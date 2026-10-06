# CloudLedger Help Centre

## KB-101 Updating your payment card
Account owners can update the card on file under Settings > Billing > Payment method. The new card is charged from the next billing date. Failed payments are retried automatically after 1, 3 and 5 days. Accounts are only suspended after 14 days of failed payments, and access is restored as soon as a payment succeeds.

## KB-102 Duplicate or unexpected charges
A duplicate charge is usually a pending authorisation that drops off within 3-5 business days. If two charges have both settled, support can refund the duplicate in full. Refunds to cards take 5-10 business days to appear. Check Settings > Billing > Invoices to compare charges against invoices.

## KB-103 Tax invoices and ABN
Tax invoices include the business name and ABN entered under Settings > Company. Update those details, then use Billing > Invoices > Download to regenerate any past invoice with the new details.

## KB-104 Switching to annual billing
Annual billing is 2 months cheaper than paying monthly. Switch under Settings > Billing > Change plan. The unused part of the current month is credited against the annual charge.

## KB-105 Refund policy
Monthly plans can be refunded for the current month if cancelled within 7 days of the charge. Annual plans can be refunded pro rata within 30 days of purchase. Charges after a confirmed cancellation are always refunded in full. Downgrades take effect at the next billing date; if the old price was charged after a downgrade, the difference is refunded.

## KB-201 Bank feed not syncing
Bank feeds disconnect when the bank requires re-consent, usually every 90 days. Go to Banking > Connections, select the bank and choose Reconnect. Missing transactions back-fill automatically within 24 hours of reconnecting.

## KB-202 PDF export issues
If invoice PDFs cut off line items, switch the template to "Compact" under Settings > Invoice templates, or reduce long item descriptions. A fix for long descriptions in the "Classic" template is in progress.

## KB-203 Mobile app crashes when attaching photos
Update to the latest app version and allow camera and photo permissions. If the app still crashes, attach the receipt from the web app and send support the device model and app version.

## KB-204 API rate limits
The API allows 60 requests per minute per account. Use the bulk endpoint /v2/invoices/batch to sync up to 100 invoices per request, and retry 429 responses with exponential backoff using the Retry-After header.

## KB-205 Service status and outages
Live service status is published at status.cloudledger.example. During an incident, updates are posted every 30 minutes. Enterprise customers receive incident notifications by email and SMS.

## KB-206 GST calculation
GST is calculated per line item at the rate set on each tax code. Check Settings > Tax codes to confirm items use "GST on Income (10%)". Invoices already sent can be corrected with a credit note and a reissued invoice.

## KB-301 Adding and removing users
Admins can invite users under Settings > Team > Invite. To remove a user, open their profile and choose Deactivate. Deactivated users cannot log in and don't count towards your seats.

## KB-302 Changing the account owner
Only the current owner can transfer ownership, under Settings > Team > Transfer ownership. If the owner has left the company, support can transfer ownership after verifying identity with two directors.

## KB-303 Two-factor authentication resets
For security, support can only reset 2FA after identity verification. The account owner can also reset 2FA for any team member under Settings > Team.

## KB-304 Password reset emails
Reset emails come from no-reply@cloudledger.example and expire after 1 hour. Check spam folders and allow-list the sender. Corporate email filters sometimes block the link.

## KB-305 Suspicious logins and account security
If you see a login you don't recognise, reset your password immediately and enable 2FA. Our security team reviews any report of changed bank details or unauthorised access within 1 hour and can lock the account to protect it.

## KB-401 Card reader delivery
Card readers ship within 2 business days and arrive in 3-7 business days by tracked post. If tracking hasn't updated for 5 business days, support can lodge a trace with the courier or send a replacement.

## KB-402 Damaged or incorrect hardware
Damaged card readers are replaced free of charge. Support sends a prepaid return label. Extra devices sent in error can be returned with the same label.

## KB-403 Changing a delivery address
Delivery addresses can be changed before the order ships. After it ships, support can ask the courier to redirect it, which may add 1-2 days.

## KB-501 Feature requests
Feature requests are logged with the product team and reviewed monthly. Popular requests are published on the public roadmap at roadmap.cloudledger.example.
