# Triaging and Acknowledging Bug Reports

This runbook defines how Meridian Cloud, Inc. handles inbound bug reports, focusing on cosmetic and low-severity issues. Support owns intake, triage, and reporter communication; confirmed defects are handed to Engineering to fix.

## Severity Scale

Support assigns a severity on receipt, independent of how urgent the reporter feels the issue is:

- **S1 — critical:** a full outage or data loss affecting many customers.
- **S2 — major:** a core workflow is broken with no reasonable workaround.
- **S3 — moderate:** a feature misbehaves but a workaround exists.
- **S4 — cosmetic or low:** visual glitches, minor UI inconsistencies, or small annoyances that do not block work.

Most inbound reports are S3 or S4. S1 and S2 issues are escalated immediately to Engineering under the P0 and P1 response targets.

## Information Needed to Reproduce

A useful bug report includes: clear steps to reproduce, what the reporter expected versus what actually happened, the browser and operating system (or mobile app version), a screenshot or screen recording when the issue is visual, any on-screen error code, and the account or workspace ID. When a report is missing reproduction steps, Support asks one focused follow-up rather than closing it, since irreproducible reports cannot be fixed.

## What the Reporter Is Told, and When

Every reporter receives an acknowledgement within Support's standard SLA — one business day for S3 and two business days for S4. The acknowledgement confirms the report was received, states the assigned severity in plain terms, and, for cosmetic issues, explains that the fix will be scheduled into a future release rather than patched immediately. Support does not promise a specific fix date for S3 and S4 items, because these are batched into regular release cycles. Reporters are updated again when the fix ships.

## Handoff to Engineering

Reproducible S3 and S4 bugs are logged in the engineering tracker with all reproduction details attached, then prioritized by Engineering against the release schedule.

## Ownership

Support owns triage and communication; Engineering owns the fix.
