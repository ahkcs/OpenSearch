/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics.planner.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.AggregateCall;
import org.opensearch.analytics.planner.PlannerContext;
import org.opensearch.analytics.planner.rel.AggregateMode;
import org.opensearch.analytics.planner.rel.OpenSearchAggregate;
import org.opensearch.analytics.planner.rel.OpenSearchConvention;

/**
 * Volcano CBO rule that splits an {@link OpenSearchAggregate} into
 * PARTIAL + FINAL when the input is partitioned.
 *
 * <p>Requests SINGLETON distribution on the partial output, letting Volcano's
 * trait enforcement (via {@code ExpandConversionRule} + {@code OpenSearchDistributionTraitDef})
 * automatically insert an {@code OpenSearchExchangeReducer}.
 *
 * <p>Decomposition responsibilities (post-refactor):
 * <ul>
 *   <li><b>Multi-field primitive decomposition</b> (AVG / STDDEV / VAR) is handled by
 *       {@link OpenSearchAggregateReduceRule} during HEP marking — before this rule runs.
 *       Volcano sees an already-reduced inner aggregate with primitive SUM/COUNT calls
 *       and a Project on top.</li>
 *   <li><b>Single-field cases</b> (pass-through SUM/MIN/MAX, function-swap COUNT→SUM at
 *       FINAL, engine-native APPROX_COUNT_DISTINCT sketch merge) are handled by
 *       {@code AggregateDecompositionResolver} after this split rule runs, reading
 *       {@link org.opensearch.analytics.spi.AggregateFunction#intermediateFields()}
 *       as the sole source of truth.</li>
 * </ul>
 *
 * <p>This rule's own contract is purely structural: SINGLE → FINAL(Exchange(PARTIAL(child))).
 * It does not rewrite aggregate calls.
 *
 * @opensearch.internal
 */
public class OpenSearchAggregateSplitRule extends RelOptRule {

    private final PlannerContext context;

    public OpenSearchAggregateSplitRule(PlannerContext context) {
        super(operand(OpenSearchAggregate.class, operand(RelNode.class, any())), "OpenSearchAggregateSplitRule");
        this.context = context;
    }

    @Override
    public boolean matches(RelOptRuleCall call) {
        OpenSearchAggregate aggregate = call.rel(0);
        if (aggregate.getMode() != AggregateMode.SINGLE) {
            return false;
        }
        // percentile_approx is a 2-arg aggregate (field, percent) whose FINAL phase needs
        // (tdigest_state, percent_literal). AggregateDecompositionResolver's single-field
        // rewrite paths only produce a single-arg FINAL call, which yields a malformed
        // OpenSearchAggregate row type and trips Calcite's RelMdRowType equivalence check
        // ("Type mismatch: rel rowtype: RecordType(BIGINT p50, BIGINT p50_0) NOT NULL,
        // equiv rowtype: RecordType(INTEGER bucket, BIGINT p50)"). Until the resolver gains
        // support for engine-native merges that preserve a literal/passthrough operand,
        // skip the split so percentile runs as a single-stage SINGLE aggregate at the
        // coordinator. Other aggCalls in the same Aggregate (SUM, AVG, etc.) inherit the
        // single-stage execution; this is a conservative correctness-over-perf trade-off
        // for a small class of queries.
        for (AggregateCall aggCall : aggregate.getAggCallList()) {
            if (isPercentileApprox(aggCall)) {
                return false;
            }
        }
        return true;
    }

    private static boolean isPercentileApprox(AggregateCall aggCall) {
        return "PERCENTILE_APPROX".equalsIgnoreCase(aggCall.getAggregation().getName());
    }

    @Override
    public void onMatch(RelOptRuleCall call) {
        OpenSearchAggregate aggregate = call.rel(0);
        RelNode child = call.rel(1);

        // Partial aggregate: runs on each partition, keeps input's traits
        RelTraitSet partialTraits = child.getTraitSet().replace(OpenSearchConvention.INSTANCE);
        OpenSearchAggregate partial = new OpenSearchAggregate(
            aggregate.getCluster(),
            partialTraits,
            child,
            aggregate.getGroupSet(),
            aggregate.getGroupSets(),
            aggregate.getAggCallList(),
            AggregateMode.PARTIAL,
            aggregate.getViableBackends()
        );

        // Request SINGLETON distribution — Volcano inserts Exchange automatically
        RelTraitSet singletonTraits = partial.getTraitSet().replace(context.getDistributionTraitDef().singleton());
        RelNode gathered = convert(partial, singletonTraits);

        // Final aggregate: merges partial states at coordinator
        OpenSearchAggregate finalAggregate = new OpenSearchAggregate(
            aggregate.getCluster(),
            singletonTraits,
            gathered,
            aggregate.getGroupSet(),
            aggregate.getGroupSets(),
            aggregate.getAggCallList(),
            AggregateMode.FINAL,
            aggregate.getViableBackends()
        );

        call.transformTo(finalAggregate);
    }
}
