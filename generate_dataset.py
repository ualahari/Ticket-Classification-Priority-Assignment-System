import random
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

technical = [
    "Internet is not working",
    "Website is very slow",
    "Application keeps crashing",
    "Unable to connect to server",
    "Error while uploading files",
    "System timeout issue"
]

billing = [
    "Incorrect billing amount",
    "Charged twice for subscription",
    "Need refund for last payment",
    "Invoice not generated",
    "Payment failed but money deducted"
]

account = [
    "Unable to login to account",
    "Password reset not working",
    "Account locked",
    "Email verification failed",
    "Username not recognized"
]

general = [
    "Thank you for the great service",
    "How to upgrade my plan?",
    "Need information about features",
    "Is customer support available 24/7?",
    "General inquiry"
]

data = []

for _ in range(10000):
    category = random.choice(["Technical", "Billing", "Account", "General"])

    if category == "Technical":
        text = random.choice(technical)
    elif category == "Billing":
        text = random.choice(billing)
    elif category == "Account":
        text = random.choice(account)
    else:
        text = random.choice(general)

    data.append([text, category])

df = pd.DataFrame(data, columns=["text", "category"])
df.to_csv("data/tickets.csv", index=False)

print("✅ Dataset created successfully at data/tickets.csv")
