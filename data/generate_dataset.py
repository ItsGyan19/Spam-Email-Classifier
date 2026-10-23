"""
Dataset Generator for Spam Email Classifier.
Creates a comprehensive, diverse dataset of emails with labels (ham/spam)
and sub-topic annotations (phishing, lottery, marketing, personal, work, etc.).
"""

import csv
import random
from pathlib import Path

# Fix random seed for reproducibility
random.seed(42)

# Templates for synthetic realistic ham (legitimate) emails
HAM_TEMPLATES = [
    # Work & Meetings
    ("work", "Hi team, please find attached the weekly sprint report. Let me know if you have questions.", "Weekly Sprint Report"),
    ("work", "Can we reschedule our sync from 2 PM to 3:30 PM today? Let me know if that works.", "Meeting Reschedule"),
    ("work", "The quarterly review slides are ready for review. Please review slides 12 to 18 before Friday.", "Q3 Review Slides"),
    ("work", "Thanks for your help debugging the API gateway yesterday. The deployment went smoothly.", "Deployment Success"),
    ("work", "Reminder: All-hands meeting scheduled for tomorrow at 10 AM in Conference Room B and on Zoom.", "All-Hands Tomorrow"),
    ("work", "Could you review my pull request #452 when you get a chance? It fixes the database query timeout.", "PR Review Request #452"),
    ("work", "Project status update: We reached milestone 2 ahead of deadline. Great job everyone!", "Milestone 2 Reached"),
    ("work", "Here is the summary of action items from today's design sprint with the UX team.", "Design Sprint Action Items"),
    ("work", "Please submit your expense reports by the end of this month to ensure timely processing.", "Expense Report Reminder"),
    ("work", "The code review session is moved to next Tuesday due to the company holiday.", "Code Review Rescheduled"),
    ("work", "Attached is the contract draft for the vendor renewal. Please check clauses 4 and 8.", "Vendor Contract Draft"),
    ("work", "Good morning, the staging environment has been updated with the latest backend release.", "Staging Environment Updated"),
    ("work", "Feedback on the architecture design: I recommend using Redis caching for session data.", "Architecture RFC Feedback"),
    ("work", "Please update your Jira tickets before the sprint retrospective this afternoon.", "Sprint Retro Prep"),
    ("work", "We are organizing a welcome lunch for the new software engineering interns this Friday.", "Intern Welcome Lunch"),

    # Personal & Casual
    ("personal", "Hey! Are we still on for dinner this Saturday? Let me know what restaurant you prefer.", "Dinner Saturday"),
    ("personal", "Happy birthday! Hope you have a fantastic day celebrating with family and friends.", "Happy Birthday!"),
    ("personal", "Thanks for letting me borrow your camping gear last weekend. It was a great trip!", "Camping Gear Return"),
    ("personal", "Just checked out the book you recommended yesterday. Really enjoying chapter 3 so far.", "Book Recommendation"),
    ("personal", "Mom mentioned you might be visiting next month. Let me know the flight dates so I can pick you up.", "Upcoming Visit"),
    ("personal", "Hey, do you still have the recipe for that homemade pasta sauce you cooked last week?", "Pasta Recipe"),
    ("personal", "Here are the photos from our hike in the national park last Sunday. Enjoy!", "Hike Photos"),
    ("personal", "Can you water my plants this weekend while I am out of town? The keys are under the mat.", "Quick Favor"),
    ("personal", "Let's catch up over coffee sometime next week. Let me know your available days.", "Coffee Catchup"),
    ("personal", "Congratulations on your new job offer! So excited for your next chapter.", "Congrats!"),

    # Legitimate transactional & notifications
    ("notification", "Your flight booking #AA9482 is confirmed. Departure: New York (JFK) at 08:30 AM.", "Flight Confirmation AA9482"),
    ("notification", "Your package from BooksOnline has shipped with tracking number 9400111202. Expected Friday.", "Package Shipped"),
    ("notification", "Monthly electricity bill is ready. Amount due: $78.45 by November 15. Auto-pay is enabled.", "Utility Bill Ready"),
    ("notification", "Your doctor appointment with Dr. Smith is confirmed for Thursday at 11:00 AM.", "Appointment Confirmation"),
    ("notification", "GitHub security alert: Dependabot detected 0 vulnerabilities in your repository.", "GitHub Security Summary"),
    ("notification", "Your library books are due in 3 days. Log in to your library account to renew online.", "Library Book Due Reminder"),
    ("notification", "Receipt for your recent ride on Oct 22: $14.20 paid with Apple Pay. Thank you for riding.", "Trip Receipt"),
    ("notification", "Your hotel reservation at Grand Hyatt has been confirmed for 3 nights checking in Nov 10.", "Hotel Reservation Confirmed"),
    ("notification", "Your subscription renewal receipt for CloudDrive 100GB: $2.99 charged to card ending in 4102.", "Subscription Receipt"),
    ("notification", "Your dental cleaning is scheduled for next Monday at 9:00 AM. Call us to reschedule.", "Dental Appointment"),

    # Newsletters & Educational
    ("newsletter", "This week in Python: Release of Python 3.13, new JIT compiler features, and performance tips.", "Python Weekly Issue #620"),
    ("newsletter", "Data Science Digest: Understanding Gradient Boosting and Transformer attention mechanisms.", "Data Science Digest #142"),
    ("newsletter", "Local community center calendar: Upcoming pottery workshops, book club meetings, and yoga classes.", "Community Center Newsletter"),
    ("newsletter", "University seminar: Guest lecture on deep learning in healthcare this Thursday at 4 PM in Hall A.", "CS Seminar Invitation"),
    ("newsletter", "Research Paper Roundup: Self-supervised learning in computer vision and natural language processing.", "AI Research Roundup"),
]

# Variations to expand ham emails
HAM_SUBJECT_PREFIXES = ["", "Re: ", "Fwd: ", "[Internal] ", "[Update] "]
HAM_GREETINGS = ["Hi,", "Hello,", "Hey there,", "Good morning,", "Dear team,", "Hi everyone,"]
HAM_SIGN_OFFS = ["Best regards,\nAlex", "Thanks,\nSarah", "Cheers,\nDavid", "Sincerely,\nTeam", "Best,\nMichael"]

# Templates for realistic spam emails across multiple malicious topics
SPAM_TEMPLATES = [
    # Phishing & Account Verification Scams
    ("phishing", "URGENT: Your PayPal account has been temporarily restricted due to suspicious login attempts. Click here to verify your identity immediately: http://paypal-verify-account-security.com", "Urgent: Account Suspended!"),
    ("phishing", "Security Alert: Someone accessed your Bank of America online banking from an unrecognized device in Russia. Confirm your password here: http://secure-boa-login-verify.net", "Security Alert: Unauthorized Access"),
    ("phishing", "Dear user, your email mailbox quota has exceeded 98%. You will stop receiving emails in 24 hours unless you upgrade your storage here: http://webmail-storage-upgrade.com/login", "Notice: Email Storage Quota Exceeded"),
    ("phishing", "IRS Notice: You have an unclaimed federal tax refund of $3,450. Submit your SSN and bank details within 48 hours to deposit funds: http://irs-refund-claim-portal.org", "Official IRS Tax Refund Notification"),
    ("phishing", "Apple Support: Your Apple ID was used to purchase iPhone 15 Pro for $1,199. If this was not you, cancel transaction now: http://appleid-cancel-dispute.me", "Apple ID Purchase Notification: Dispute Now"),
    ("phishing", "Netflix Membership Alert: Your payment method failed and account is on hold. Update billing information to resume streaming: http://netflix-billing-update-service.info", "Netflix: Payment Declined - Update Info"),
    ("phishing", "Amazon Security: Suspicious purchase detected on your account for $850 Gift Card. Call customer support immediately or verify: http://amazon-customer-verify-portal.cc", "Amazon: Unusual Activity on Your Account"),
    ("phishing", "Microsoft Office 365: Your password expires in 2 hours. Keep current password by validating your corporate account now: http://office365-pass-renew.com", "Office365: Password Expiration Notice"),

    # Financial / Lottery / Inheritance Scams
    ("lottery", "CONGRATULATIONS! You have been selected as the grand winner of $2,500,000 in the International EuroMillions Lottery! Reply with your full name, age, and phone number to claim.", "CONGRATULATIONS!! You Won $2,500,000 USD!!"),
    ("lottery", "Dear Friend, I am Barrister Williams. My late client left an unclaimed sum of $14.5 Million in a vault. I need your partnership to transfer these funds. You will receive 40% share.", "CONFIDENTIAL PARTNERSHIP PROPOSAL: $14.5M"),
    ("lottery", "You are the lucky winner of our annual cash sweepstakes award! Claim your $50,000 Walmart Gift Card today! Click here to claim your reward: http://claim-free-gift-card-500.com", "Claim Your $50,000 Gift Card TODAY"),
    ("lottery", "ATTENTION: Central Bank of Nigeria wire transfer approved. $5,000,000 ready for wire transfer to your personal bank account. Send your bank details immediately to release funds.", "WIRE TRANSFER RELEASE NOTIFICATION #994"),
    ("lottery", "Dear beneficiary, your ATM Master Card valued at $750,000 is ready for courier delivery to your residential address. Pay $150 delivery fee to proceed.", "Your ATM Master Card Delivery Notice"),
    ("lottery", "Final Notice: Your name appeared in the global sweepstakes prize draw! Cash payout of $100,000 awaiting claim. Call 1-800-CLAIM-NOW or reply immediately.", "FINAL NOTICE: $100,000 Prize Draw Winner"),

    # Commercial / Aggressive Marketing / Fake Discounts
    ("marketing", "LOSE 30 LBS IN 2 WEEKS! Miracle keto diet supplement doctor recommended! 70% off today only! Buy now: http://miracle-keto-fatburn.com Free shipping worldwide!", "Lose 30 lbs in 14 days - 70% OFF!"),
    ("marketing", "Cheap Viagra, Cialis, and Levitra pills online without prescription! Fast discreet overnight delivery guaranteed! Order now: http://rx-online-pharma-direct.biz", "Buy cheap pills online - No prescription"),
    ("marketing", "HOT CASINO BONUSES! Claim 500 Free Spins + $1,000 Welcome Bonus at MegaJackpot Casino! No deposit required! Sign up today: http://megajackpot-casino-bonus.net", "500 Free Spins + $1,000 Bonus!"),
    ("marketing", "Pre-approved for $50,000 personal loan! Bad credit ok! 0% APR for first 12 months! Fast cash deposited in your account within 1 hour: http://instant-cash-loans-fast.org", "Pre-approved $50,000 Loan - No Credit Check"),
    ("marketing", "Lowest mortgage refinancing rates in history! Lower your monthly payment by up to $600/month! Check your free rate quote in 60 seconds: http://refinance-home-low-rates.info", "Refinance your mortgage - Save $600/mo"),
    ("marketing", "Luxury Rolex, Omega, and Cartier watches at 90% discount! 100% genuine replica quality with Swiss movements! Shop our flash sale: http://luxury-replicas-outlet.shop", "Flash Sale: Luxury Watches 90% OFF"),

    # Crypto & Work-from-Home / Get-rich-quick scams
    ("crypto_scam", "EARN $500 TO $2,000 PER DAY WORKING FROM HOME! No experience needed! Start immediately with just 1 hour per day! Visit: http://automated-income-system.biz", "Make $1,500/day working from home!"),
    ("crypto_scam", "Exclusive Bitcoin AI Trading Bot makes 450% guaranteed return every 7 days! Join over 10,000 millionaires! Register your free crypto wallet now: http://btc-ai-millionaire-robot.io", "Guaranteed 450% ROI with Bitcoin AI Bot"),
    ("crypto_scam", "URGENT CRYPTO AIRDROP: Claim 5,000 FREE Ethereum tokens before deadline midnight! Connect your MetaMask or Trust wallet here: http://eth-airdrop-claim-portal.xyz", "5,000 ETH Airdrop Available - Claim Now"),
    ("crypto_scam", "Secret investment opportunity: Multiply your money 10x in 30 days! Risk-free foreign exchange trading platform. Act now before registration closes!", "10x Your Money in 30 Days - Secret Strategy"),
    ("crypto_scam", "Congratulations! Your email address was randomly selected to receive 0.5 Bitcoin ($32,000). Click link to withdraw funds to your wallet: http://claim-free-bitcoin-drop.co", "You Received 0.5 BTC ($32,000)"),
]

SPAM_PREFIXES = ["URGENT: ", "ACT NOW! ", "CONGRATULATIONS: ", "ALERT: ", "OFFICIAL NOTICE: ", ""]
SPAM_CALL_TO_ACTIONS = [
    " Act immediately to claim before this offer expires!",
    " Do not ignore this email. Limited time only!",
    " Click the link below right now or your account will be permanently closed.",
    " 100% free guaranteed. No risk, reply with details today!",
    " Immediate response required to avoid cancellation.",
    " Wire transfer will be revoked if not claimed in 24 hours."
]

def generate_samples():
    records = []
    
    # 1. Base HAM generation with variations
    for _ in range(35):  # creates ~35 * 35 = 1225 ham samples
        for topic, body, subject in HAM_TEMPLATES:
            greeting = random.choice(HAM_GREETINGS)
            sign_off = random.choice(HAM_SIGN_OFFS)
            prefix = random.choice(HAM_SUBJECT_PREFIXES)
            
            # Subtle natural noise/variations
            text = f"{greeting}\n\n{body}\n\n{sign_off}"
            sub = f"{prefix}{subject}"
            full_content = f"Subject: {sub}\n\n{text}"
            
            records.append({
                "label": "ham",
                "topic": topic,
                "subject": sub,
                "body": text,
                "content": full_content
            })
            
    # 2. Base SPAM generation with variations
    for _ in range(48):  # creates ~48 * 25 = 1200 spam samples
        for topic, body, subject in SPAM_TEMPLATES:
            prefix = random.choice(SPAM_PREFIXES)
            cta = random.choice(SPAM_CALL_TO_ACTIONS)
            
            sub = f"{prefix}{subject}"
            full_body = f"{body}{cta}"
            full_content = f"Subject: {sub}\n\n{full_body}"
            
            records.append({
                "label": "spam",
                "topic": topic,
                "subject": sub,
                "body": full_body,
                "content": full_content
            })
            
    # Shuffle records
    random.shuffle(records)
    return records

def main():
    data_dir = Path(__file__).resolve().parent
    output_path = data_dir / "emails.csv"
    
    samples = generate_samples()
    print(f"Generating {len(samples)} diverse email samples...")
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "topic", "subject", "body", "content"])
        writer.writeheader()
        writer.writerows(samples)
        
    ham_count = sum(1 for s in samples if s["label"] == "ham")
    spam_count = sum(1 for s in samples if s["label"] == "spam")
    print(f"Successfully saved to {output_path}")
    print(f"Total samples: {len(samples)} | Ham: {ham_count} | Spam: {spam_count}")

if __name__ == "__main__":
    main()

