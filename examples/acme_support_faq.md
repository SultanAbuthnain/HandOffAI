# Acme Cloud Platform — Customer Support Knowledge Base & FAQ

## 1. Plans & Pricing
Acme Cloud offers three subscription tiers billed monthly or annually (annual billing saves 20%):
- **Starter Plan ($29/month):** Up to 5 team members, 10,000 monthly API requests, email support, and 7-day log retention.
- **Pro Plan ($99/month):** Up to 25 team members, 250,000 monthly API requests, priority chat support, SSO (Google & Microsoft), and 30-day log retention.
- **Enterprise Plan ($399/month):** Unlimited team members, 2,000,000 monthly API requests, 99.95% uptime SLA, SAML/SCIM provisioning, dedicated Slack channel, and 365-day audit log retention.

## 2. Refund & Cancellation Policy
- **30-Day Money-Back Guarantee:** New customers on any monthly or annual plan may request a full refund within 30 calendar days of their initial purchase.
- **How to Cancel:** Account owners can cancel anytime under **Settings → Billing → Cancel Subscription**. Access continues until the end of the current billing cycle.
- **Prorated Credits:** Downgrading mid-cycle generates automatic prorated credit applied to future invoices.

## 3. Human Resources (HR) & Directory Integrations
- Acme Cloud integrates natively with popular Human Resources Information Systems (HRIS) including **Workday**, **BambooHR**, **Rippling**, and **Gusto** via SCIM 2.0 on the Enterprise plan.
- When an employee is onboarded or offboarded in your Human Resources portal, their Acme Cloud seat and role permissions are provisioned or revoked automatically within 5 minutes.
- Document workflows also support an optional human review approval gate before automated publishing.

## 4. API Rate Limits & Authentication
- Authenticate all API calls using an `Authorization: Bearer <ACME_API_KEY>` header over HTTPS (`https://api.acme.example.com/v1`).
- Default burst limit: **120 requests per minute** on Starter, **600 requests per minute** on Pro, and **3,000 requests per minute** on Enterprise.
- Exceeding rate limits returns HTTP `429 Too Many Requests` with a `Retry-After` header indicating how many seconds to wait.

## 5. Support Hours & SLA Response Times
- **AI Support Assistant:** Available 24/7 for instant documentation and policy answers.
- **Live Human Support Engineers:** Available Monday–Friday, 08:00–20:00 UTC.
- **Target First-Response SLA:**
  - Enterprise critical incidents: **15 minutes**
  - Pro plan inquiries: **2 hours**
  - Starter plan inquiries: **12 business hours**
