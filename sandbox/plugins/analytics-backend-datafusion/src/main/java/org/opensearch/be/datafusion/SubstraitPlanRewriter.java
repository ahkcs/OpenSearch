/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.be.datafusion;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

import io.substrait.expression.AggregateFunctionInvocation;
import io.substrait.expression.Expression;
import io.substrait.expression.FieldReference;
import io.substrait.expression.FunctionArg;
import io.substrait.expression.ImmutableAggregateFunctionInvocation;
import io.substrait.expression.ImmutableExpression;
import io.substrait.plan.Plan;
import io.substrait.relation.Aggregate;
import io.substrait.relation.ExpressionCopyOnWriteVisitor;
import io.substrait.relation.ImmutableAggregate;
import io.substrait.relation.ImmutableMeasure;
import io.substrait.relation.Project;
import io.substrait.relation.Rel;
import io.substrait.relation.RelCopyOnWriteVisitor;
import io.substrait.util.EmptyVisitationContext;

/**
 * Single-pass post-processor for Substrait plans before serialization to protobuf.
 *
 * <p>Applies two kinds of rewrites:
 * <ul>
 *   <li><b>Rel-level</b> — structural changes like table name stripping, handled by
 *       {@link RelCopyOnWriteVisitor} overrides.</li>
 *   <li><b>Expression-level</b> — literal/type fixes handled by
 *       {@link ExpressionCopyOnWriteVisitor} overrides. Adding a new expression rewrite
 *       only requires overriding the corresponding {@code visit} method.</li>
 * </ul>
 *
 * @opensearch.internal
 */
class SubstraitPlanRewriter {

    private SubstraitPlanRewriter() {}

    static Plan rewrite(Plan plan) {
        PlanRelVisitor visitor = new PlanRelVisitor();

        List<Plan.Root> roots = new ArrayList<>();
        for (Plan.Root root : plan.getRoots()) {
            Optional<Rel> modified = root.getInput().accept(visitor, null);
            roots.add(modified.isPresent() ? Plan.Root.builder().from(root).input(modified.get()).build() : root);
        }
        return Plan.builder().from(plan).roots(roots).build();
    }

    /**
     * Rel-level visitor. Handles structural rewrites (table name stripping) and delegates
     * expression rewrites to {@link PlanExpressionVisitor}.
     */
    private static class PlanRelVisitor extends RelCopyOnWriteVisitor<RuntimeException> {

        private final PlanExpressionVisitor expressionVisitor = new PlanExpressionVisitor(this);

        // Rewrite expressions inside filter conditions
        @Override
        public Optional<Rel> visit(io.substrait.relation.Filter filter, EmptyVisitationContext ctx) {
            Optional<Rel> newInput = filter.getInput().accept(this, ctx);
            Optional<Expression> rewritten = filter.getCondition().accept(expressionVisitor, ctx);
            if (newInput.isEmpty() && rewritten.isEmpty()) return Optional.empty();
            return Optional.of(
                io.substrait.relation.Filter.builder()
                    .from(filter)
                    .input(newInput.orElse(filter.getInput()))
                    .condition(rewritten.orElse(filter.getCondition()))
                    .build()
            );
        }

        /**
         * For each {@code approx_percentile_cont} measure, lift the percent column-ref to an
         * inline {@link Expression.FP64Literal} by tracing it back to the upstream Project's
         * expression at the referenced column index. DataFusion's
         * {@code APPROX_PERCENTILE_CONT} planner requires the percent argument to be a
         * literal — not a column reference — and isthmus 0.89.1's
         * {@code WrappedAggregateCall.getOperands} drops {@code AggregateCall.rexList}
         * silently, so we have to thread the literal through argList and inline it here.
         */
        @Override
        public Optional<Rel> visit(Aggregate aggregate, EmptyVisitationContext ctx) {
            Optional<Rel> newInput = aggregate.getInput().accept(this, ctx);
            Rel resolvedInput = newInput.orElse(aggregate.getInput());
            List<Expression> projectExprs = (resolvedInput instanceof Project p) ? p.getExpressions() : null;

            List<Aggregate.Measure> rewrittenMeasures = new ArrayList<>(aggregate.getMeasures().size());
            boolean measuresChanged = false;
            for (Aggregate.Measure measure : aggregate.getMeasures()) {
                AggregateFunctionInvocation func = measure.getFunction();
                if (projectExprs != null && "approx_percentile_cont".equalsIgnoreCase(func.declaration().name())) {
                    Optional<AggregateFunctionInvocation> rewritten = liftPercentileLiteral(func, projectExprs);
                    if (rewritten.isPresent()) {
                        rewrittenMeasures.add(ImmutableMeasure.builder().from(measure).function(rewritten.get()).build());
                        measuresChanged = true;
                        continue;
                    }
                }
                rewrittenMeasures.add(measure);
            }

            if (newInput.isEmpty() && !measuresChanged) {
                return Optional.empty();
            }
            return Optional.of(ImmutableAggregate.builder().from(aggregate).input(resolvedInput).measures(rewrittenMeasures).build());
        }

        private Optional<AggregateFunctionInvocation> liftPercentileLiteral(
            AggregateFunctionInvocation func,
            List<Expression> projectExprs
        ) {
            List<FunctionArg> args = func.arguments();
            // Expect exactly (field_ref, percent_arg) — see PlannerImpl.rewritePercentileApprox.
            if (args.size() != 2) return Optional.empty();
            FunctionArg percentArg = args.get(1);
            if (!(percentArg instanceof FieldReference fieldRef)) return Optional.empty();
            Integer fieldIdx = simpleStructFieldIndex(fieldRef);
            if (fieldIdx == null || fieldIdx < 0 || fieldIdx >= projectExprs.size()) return Optional.empty();
            Expression projectedExpr = projectExprs.get(fieldIdx);
            Expression literal = unwrapToLiteral(projectedExpr);
            if (literal == null) return Optional.empty();
            List<FunctionArg> newArgs = new ArrayList<>(args);
            newArgs.set(1, literal);
            return Optional.of(ImmutableAggregateFunctionInvocation.builder().from(func).arguments(newArgs).build());
        }

        /**
         * For a simple root struct-field reference (no nested element/map traversal), returns
         * the field index. Returns {@code null} for any other shape.
         */
        private static Integer simpleStructFieldIndex(FieldReference ref) {
            if (!ref.isSimpleRootReference() || ref.segments().size() != 1) return null;
            FieldReference.ReferenceSegment seg = ref.segments().get(0);
            if (seg instanceof FieldReference.StructField sf) {
                return sf.offset();
            }
            return null;
        }

        /**
         * Walks {@code expr} through {@link Expression.Cast} wrappers — Calcite often emits a
         * type-pinning {@code CAST(0.5E0:DOUBLE):DOUBLE} after constant-folding the rescale —
         * and returns the inner {@link Expression.Literal} (untouched) if there is one. The
         * literal value carries the percent fraction; the wrapping CAST is redundant for
         * DataFusion's percentile-arg consumer, which just needs a scalar literal.
         */
        private static Expression unwrapToLiteral(Expression expr) {
            Expression cur = expr;
            while (cur instanceof Expression.Cast cast) {
                cur = cast.input();
            }
            if (cur instanceof Expression.Literal) {
                return cur;
            }
            return null;
        }
    }

    /**
     * Expression-level visitor. Override a {@code visit} method to add a new rewrite.
     * The base class handles recursion into function arguments, casts, if-then, etc.
     */
    private static class PlanExpressionVisitor extends ExpressionCopyOnWriteVisitor<RuntimeException> {

        PlanExpressionVisitor(PlanRelVisitor relVisitor) {
            super(relVisitor);
        }

        // Isthmus hardcodes timestamp literals to precision 6 (microseconds).
        // Parquet stores Timestamp(MILLISECOND), so convert to precision 3.
        @Override
        public Optional<Expression> visit(Expression.PrecisionTimestampLiteral pts, EmptyVisitationContext ctx) {
            if (pts.precision() != 3) {
                return Optional.of(
                    ImmutableExpression.PrecisionTimestampLiteral.builder()
                        .value(toMillis(pts.value(), pts.precision()))
                        .precision(3)
                        .nullable(pts.nullable())
                        .build()
                );
            }
            return Optional.empty();
        }
    }

    private static long toMillis(long value, int precision) {
        return switch (precision) {
            case 0 -> value * 1000L;
            case 6 -> TimeUnit.MICROSECONDS.toMillis(value);
            case 9 -> TimeUnit.NANOSECONDS.toMillis(value);
            default -> throw new IllegalArgumentException(
                "Unsupported timestamp precision: " + precision + ". Expected 0 (seconds), 6 (micros), or 9 (nanos)."
            );
        };
    }
}
