# Understanding Invoice Line Items

This runbook explains how a Meridian Cloud, Inc. invoice is structured so agents can interpret any charge a customer questions. Billing Ops owns invoice generation, which runs through Stripe on the customer's billing anniversary.

## Base Subscription

Every invoice opens with the base subscription line for the active plan: Starter at $12 per seat per month, Business at $28 per seat per month, or a negotiated Enterprise rate starting at $45 per seat per month (50-seat minimum). Annual customers see the full-year amount billed once, reflecting the two-months-free discount versus monthly billing. Monthly customers see a single month.

## Per-Seat Charges and Mid-Cycle Changes

Seat count is billed on the plan's per-seat rate. When an account adds seats mid-cycle, Meridian charges a prorated amount for the remainder of the current period, shown as a separate "Additional seats (prorated)" line. Removing seats does not trigger a refund line; the reduced count takes effect at the next renewal. A seat-count discrepancy is the most common escalation and should be routed to Billing Ops for reconciliation against the workspace admin panel.

## Processing Fees

Card and ACH payments carry no fee. Invoices paid by wire transfer or check that total under $2,000 incur a flat $15 offline-payment processing fee, listed as "Manual payment processing." This fee is waived automatically once the account switches to card or ACH autopay.

## Proration

Proration appears whenever a change lands mid-period: upgrades, added seats, or plan-tier switches. Each prorated line shows the daily rate and the number of days remaining, so the math is auditable.

## Taxes

Sales tax is calculated at the customer's billing address through Avalara and appears as a distinct line beneath the subtotal. Tax-exempt organizations must file a valid exemption certificate with Billing Ops before renewal; exemptions are not applied retroactively to invoices already issued.

## Ownership

Billing Ops is the system of record for all invoice questions and adjustments.
