/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics.planner;

import org.apache.calcite.plan.Contexts;
import org.apache.calcite.plan.ConventionTraitDef;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.plan.hep.HepMatchOrder;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.plan.volcano.AbstractConverter;
import org.apache.calcite.plan.volcano.VolcanoPlanner;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.rules.ReduceExpressionsRule;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.analytics.planner.rel.OpenSearchDistributionTraitDef;
import org.opensearch.analytics.planner.rules.OpenSearchAggregateReduceRule;
import org.opensearch.analytics.planner.rules.OpenSearchAggregateRule;
import org.opensearch.analytics.planner.rules.OpenSearchAggregateSplitRule;
import org.opensearch.analytics.planner.rules.OpenSearchFilterRule;
import org.opensearch.analytics.planner.rules.OpenSearchProjectRule;
import org.opensearch.analytics.planner.rules.OpenSearchSortRule;
import org.opensearch.analytics.planner.rules.OpenSearchTableScanRule;
import org.opensearch.analytics.planner.rules.OpenSearchUnionRule;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Central planner for the Analytics Plugin.
 *
 * <p>Two phases:
 * <ol>
 *   <li>HepPlanner (RBO): converts LogicalXxx → OpenSearchXxx with backend
 *       assignment, predicate annotation, and distribution traits.</li>
 *   <li>VolcanoPlanner (CBO): requests SINGLETON at root (coordinator must
 *       gather all results). Split rule fires on aggregates, Volcano inserts
 *       exchanges via trait enforcement where distribution mismatches.</li>
 * </ol>
 *
 * <p>TODO: eliminate copyToCluster — have frontends create RelNodes with Volcano cluster.
 * <p>TODO: DAG construction (cut at exchange boundaries, build stage tree)
 * <p>TODO: Per-stage plan forking (multiple plan generation)
 * <p>TODO: Fragment conversion (backend.getFragmentConvertor())
 * <p>TODO: Join strategy selection, sort removal via CBO
 *
 * @opensearch.internal
 */
public class PlannerImpl {

    private static final Logger LOGGER = LogManager.getLogger(PlannerImpl.class);

    public static RelNode createPlan(RelNode rawRelNode, PlannerContext context) {
        try {
            return markAndOptimize(rawRelNode, context);
        } catch (AssertionError e) {
            // Calcite's Litmus.THROW (used by RelOptUtil.eq, Aggregate.typeMatchesInferred,
            // various validators) throws AssertionError directly via Java code rather than via
            // the `assert` keyword, so JVM -da doesn't gate them. If one fires inside a search
            // thread, OpenSearchUncaughtExceptionHandler exits the cluster JVM. Convert to
            // IllegalStateException so the analytics-engine error path treats it as a normal
            // per-query failure (HTTP 500 with a bucketable message) instead of taking down
            // the cluster.
            throw new IllegalStateException("Analytics-engine planner rejected the plan: " + e.getMessage(), e);
        }
    }

    /**
     * Phase 1 (RBO marking) + Phase 2 (CBO exchange insertion).
     * Package-private so planner rule tests can inspect the marked+optimized tree.
     */
    public static RelNode markAndOptimize(RelNode rawRelNode, PlannerContext context) {
        LOGGER.info("Input RelNode:\n{}", RelOptUtil.toString(rawRelNode));

        // Rewrite PPL's `percentile_approx(field, percent, sqlflag[, compression])` aggregate
        // calls into a 2-arg shape `(field, percent_fraction)`. PPL injects the sqlflag as an
        // internal type-tag for its TDigest UDAF; a backend that maps the call to its own
        // native approximate-percentile implementation has no use for it. SYMBOL also isn't a
        // known FieldType, so leaving it in trips OpenSearchAggregateRule's storage-info
        // resolve. We keep BOTH the field and the percent column in argList so isthmus 0.89.1
        // emits both args; SubstraitPlanRewriter then lifts the percent column-ref to a Literal
        // post-isthmus (DataFusion requires the percent as a literal). Run at the top of
        // markAndOptimize so the SYMBOL column is gone before any rule (incl.
        // OpenSearchAggregateReduceRule) sees the plan.
        // TODO: move to a per-backend RelNode preprocess hook on AnalyticsSearchBackendPlugin
        // once that SPI exists.
        rawRelNode = rewritePercentileApprox(rawRelNode);

        // Phase 1a: Pre-marking logical optimizations (constant expression reduction)
        HepProgramBuilder preBuilder = new HepProgramBuilder();
        preBuilder.addMatchOrder(HepMatchOrder.ARBITRARY);
        preBuilder.addRuleCollection(
            List.of(
                new ReduceExpressionsRule.FilterReduceExpressionsRule(Filter.class, RelBuilder.proto(Contexts.empty())),
                new ReduceExpressionsRule.ProjectReduceExpressionsRule(Project.class, RelBuilder.proto(Contexts.empty()))
            )
        );
        HepPlanner prePlanner = new HepPlanner(preBuilder.build());
        prePlanner.setRoot(rawRelNode);
        RelNode afterPre = prePlanner.findBestExp();

        // Phase 1b: Aggregate-reduction — decompose AVG / STDDEV / VAR into primitive SUM/COUNT
        // (+ SUM_SQ for variance) plus a scalar LogicalProject computing the quotient. Runs as
        // its own HEP pass on plain LogicalAggregate so Calcite's type inference is clean —
        // no AGG_CALL_ANNOTATION wrappers in aggCall.rexList to propagate AVG's DOUBLE return
        // type to the derived primitive calls. Downstream the marking phase, the Volcano split
        // rule, and the AggregateDecompositionResolver see correctly-typed primitives.
        HepProgramBuilder reduceBuilder = new HepProgramBuilder();
        reduceBuilder.addMatchOrder(HepMatchOrder.BOTTOM_UP);
        reduceBuilder.addRuleInstance(new OpenSearchAggregateReduceRule());
        HepPlanner reducePlanner = new HepPlanner(reduceBuilder.build());
        reducePlanner.setRoot(afterPre);
        RelNode afterReduce = reducePlanner.findBestExp();

        // Phase 1c: Marking — convert LogicalXxx → OpenSearchXxx bottom-up
        // TODO: migrate rules from deprecated RelOptRule to RelRule<Config> once the planner
        // moves to its own Gradle module. The OpenSearch monorepo injects -proc:none globally,
        // blocking the Immutables annotation processor required by RelRule.Config sub-interfaces.
        // TODO: add SortPushdown rule here — pushes Sort below Exchange to data nodes for top-K
        // optimization.
        HepProgramBuilder markBuilder = new HepProgramBuilder();
        markBuilder.addMatchOrder(HepMatchOrder.BOTTOM_UP);
        markBuilder.addRuleCollection(
            List.of(
                new OpenSearchTableScanRule(context),
                new OpenSearchFilterRule(context),
                new OpenSearchProjectRule(context),
                new OpenSearchAggregateRule(context),
                new OpenSearchSortRule(context),
                new OpenSearchUnionRule(context)
            )
        );
        HepPlanner markingPlanner = new HepPlanner(markBuilder.build());
        markingPlanner.setRoot(afterReduce);
        RelNode marked = markingPlanner.findBestExp();

        LOGGER.info("After marking:\n{}", RelOptUtil.toString(marked));

        // Phase 2: CBO — VolcanoPlanner for trait propagation + exchange insertion
        VolcanoPlanner volcanoPlanner = new VolcanoPlanner();
        volcanoPlanner.addRelTraitDef(ConventionTraitDef.INSTANCE);
        OpenSearchDistributionTraitDef distTraitDef = context.getDistributionTraitDef();
        volcanoPlanner.addRelTraitDef(distTraitDef);
        volcanoPlanner.addRule(new OpenSearchAggregateSplitRule(context));
        volcanoPlanner.addRule(AbstractConverter.ExpandConversionRule.INSTANCE);

        RelOptCluster volcanoCluster = RelOptCluster.create(volcanoPlanner, rawRelNode.getCluster().getRexBuilder());
        volcanoCluster.setMetadataQuerySupplier(RelMetadataQuery::instance);

        // TODO: eliminate this copy
        RelNode copied = RelNodeUtils.copyToCluster(marked, volcanoCluster, distTraitDef);

        // Root must be SINGLETON — coordinator gathers all results
        volcanoPlanner.setRoot(copied);
        RelTraitSet desiredTraits = copied.getTraitSet().replace(distTraitDef.singleton());
        if (!copied.getTraitSet().equals(desiredTraits)) {
            volcanoPlanner.setRoot(volcanoPlanner.changeTraits(copied, desiredTraits));
        }
        RelNode result = volcanoPlanner.findBestExp();

        LOGGER.info("After CBO:\n{}", RelOptUtil.toString(result));
        return result;
    }

    /**
     * Walks the tree and rewrites Aggregate+Project pairs that include a
     * {@code percentile_approx(field, percent, sqlflag[, compression])} call.
     *
     * <p>Drops the SYMBOL-typed sqlflag column (and any compression hint column) from the
     * feeding Project, rescales the percent column from {@code [0,100]} to {@code [0,1]},
     * and updates the Aggregate's argList to {@code [field, percent]} skipping the dropped
     * indices. Aggregates without any percentile_approx call, or whose input isn't a Project,
     * are left unchanged.
     */
    private static RelNode rewritePercentileApprox(RelNode node) {
        boolean changed = false;
        List<RelNode> newInputs = new ArrayList<>(node.getInputs().size());
        for (RelNode input : node.getInputs()) {
            RelNode rewritten = rewritePercentileApprox(input);
            newInputs.add(rewritten);
            if (rewritten != input) {
                changed = true;
            }
        }
        RelNode current = changed ? node.copy(node.getTraitSet(), newInputs) : node;

        if (!(current instanceof Aggregate agg)) {
            return current;
        }
        boolean hasPercentile = agg.getAggCallList().stream().anyMatch(PlannerImpl::isPercentileApproxCall);
        if (!hasPercentile) {
            return current;
        }
        if (!(agg.getInput() instanceof Project project)) {
            return current;
        }
        int projectFieldCount = project.getProjects().size();
        List<RexNode> projectExprs = project.getProjects();

        // Identify columns referenced as the trailing extra args (position 2+) of any
        // percentile_approx call — those are the sqlflag + optional compression hint and
        // are candidates to drop. Track which Project indices are referenced by NON-percentile
        // users so we don't accidentally drop them. Position-1 (percent) columns are kept and
        // rescaled in-place.
        boolean[] usedByOther = new boolean[projectFieldCount];
        boolean[] usedAsPercentColumn = new boolean[projectFieldCount];
        for (AggregateCall call : agg.getAggCallList()) {
            List<Integer> args = call.getArgList();
            if (isPercentileApproxCall(call)) {
                if (!args.isEmpty() && args.get(0) >= 0 && args.get(0) < projectFieldCount) {
                    usedByOther[args.get(0)] = true; // field column — kept
                }
                if (args.size() >= 2 && args.get(1) >= 0 && args.get(1) < projectFieldCount) {
                    usedAsPercentColumn[args.get(1)] = true;
                }
            } else {
                for (int idx : args) {
                    if (idx >= 0 && idx < projectFieldCount) {
                        usedByOther[idx] = true;
                    }
                }
            }
        }
        for (int idx : agg.getGroupSet()) {
            if (idx >= 0 && idx < projectFieldCount) {
                usedByOther[idx] = true;
            }
        }
        boolean[] drop = new boolean[projectFieldCount];
        for (AggregateCall call : agg.getAggCallList()) {
            if (!isPercentileApproxCall(call)) continue;
            List<Integer> args = call.getArgList();
            for (int i = 2; i < args.size(); i++) {
                int idx = args.get(i);
                if (idx >= 0 && idx < projectFieldCount && !usedByOther[idx]) {
                    drop[idx] = true;
                }
            }
        }
        // Build the new Project. Percent columns are rescaled in place to fp64 / 100.0;
        // dropped indices are skipped; everything else passes through.
        RexBuilder rexBuilder = project.getCluster().getRexBuilder();
        RelDataTypeFactory typeFactory = rexBuilder.getTypeFactory();
        RelDataType fp64NotNull = typeFactory.createSqlType(SqlTypeName.DOUBLE);
        List<RexNode> newExprs = new ArrayList<>();
        List<String> newNames = new ArrayList<>();
        Map<Integer, Integer> remap = new HashMap<>();
        List<String> oldNames = project.getRowType().getFieldNames();
        for (int i = 0; i < projectFieldCount; i++) {
            if (drop[i]) continue;
            RexNode expr = projectExprs.get(i);
            if (usedAsPercentColumn[i] && !usedByOther[i]) {
                // Rescale percent: CAST(percent AS DOUBLE) / 100.0 — produces a literal fp64
                // value at planning time after constant fold.
                RexNode asDouble = rexBuilder.makeCast(fp64NotNull, expr, false);
                expr = rexBuilder.makeCall(
                    SqlStdOperatorTable.DIVIDE,
                    asDouble,
                    rexBuilder.makeApproxLiteral(java.math.BigDecimal.valueOf(100.0))
                );
            }
            remap.put(i, newExprs.size());
            newExprs.add(expr);
            newNames.add(oldNames.get(i));
        }
        RelNode newProject = RelBuilder.proto(Contexts.empty())
            .create(project.getCluster(), null)
            .push(project.getInput())
            .project(newExprs, newNames, true)
            .build();

        // Rebuild Aggregate calls — for percentile calls, narrow argList to [field, percent]
        // and remap indices. Other calls just remap their argList. Keep the original return
        // type since field is operand 0, ARG0_FORCE_NULLABLE still picks the field type.
        ImmutableBitSet newGroupSet = remapBitSet(agg.getGroupSet(), remap);
        List<ImmutableBitSet> newGroupSets = agg.getGroupSets().stream().map(s -> remapBitSet(s, remap)).toList();
        List<AggregateCall> newCalls = new ArrayList<>(agg.getAggCallList().size());
        for (AggregateCall call : agg.getAggCallList()) {
            List<Integer> oldArgs = call.getArgList();
            int retainCount = isPercentileApproxCall(call) ? Math.min(2, oldArgs.size()) : oldArgs.size();
            List<Integer> newArgs = new ArrayList<>(retainCount);
            for (int i = 0; i < retainCount; i++) {
                Integer remapped = remap.get(oldArgs.get(i));
                if (remapped != null) {
                    newArgs.add(remapped);
                }
            }
            ImmutableBitSet newDistinctKeys = call.distinctKeys != null ? remapBitSet(call.distinctKeys, remap) : null;
            newCalls.add(
                AggregateCall.create(
                    call.getAggregation(),
                    call.isDistinct(),
                    call.isApproximate(),
                    call.ignoreNulls(),
                    call.rexList,
                    newArgs,
                    call.filterArg >= 0 ? remap.getOrDefault(call.filterArg, -1) : call.filterArg,
                    newDistinctKeys,
                    call.collation,
                    call.getType(),
                    call.getName()
                )
            );
        }
        return agg.copy(agg.getTraitSet(), newProject, newGroupSet, newGroupSets, newCalls);
    }

    private static boolean isPercentileApproxCall(AggregateCall call) {
        SqlAggFunction op = call.getAggregation();
        return (op.getKind() == SqlKind.OTHER_FUNCTION || op.getKind() == SqlKind.OTHER)
            && "percentile_approx".equalsIgnoreCase(op.getName());
    }

    private static ImmutableBitSet remapBitSet(ImmutableBitSet bits, Map<Integer, Integer> remap) {
        ImmutableBitSet.Builder b = ImmutableBitSet.builder();
        for (int idx : bits) {
            Integer m = remap.get(idx);
            if (m != null) {
                b.set(m);
            }
        }
        return b.build();
    }
}
