import json
import os
import random
from typing import Dict, Any, List

# Ensure directories exist
APPLICATIONS_DIR = "data/applications"
RESULTS_DIR = "data/results"
os.makedirs(APPLICATIONS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_application(approval_type=None):
    """Generate application with extremely clear patterns based on approval_type"""
    if approval_type == "approve_high_score":
        # This application should be approved due to high credit score
        credit_score = random.randint(750, 850)
        credit_tier = "Excellent"
        annual_income = random.randint(50000, 70000)  # Medium income
        income_tier = "Medium"
        debt_ratio = random.uniform(0.3, 0.4)  # Medium debt
        debt_tier = "Medium"
        payment_history = "Good"
        employment_status = "Employed"
    
    elif approval_type == "approve_high_income":
        # This application should be approved due to high income
        credit_score = random.randint(680, 720)  # Good but not excellent
        credit_tier = "Good"
        annual_income = random.randint(100000, 150000)  # High income
        income_tier = "Very High"
        debt_ratio = random.uniform(0.3, 0.4)  # Medium debt
        debt_tier = "Medium"
        payment_history = "Good"
        employment_status = "Employed"
    
    elif approval_type == "approve_low_debt":
        # This application should be approved due to low debt
        credit_score = random.randint(680, 720)  # Good but not excellent
        credit_tier = "Good"
        annual_income = random.randint(50000, 70000)  # Medium income
        income_tier = "Medium"
        debt_ratio = random.uniform(0.1, 0.2)  # Very low debt
        debt_tier = "Very Low"
        payment_history = "Good"
        employment_status = "Employed"
    
    elif approval_type == "decline_low_score":
        # This application should be declined due to low score
        credit_score = random.randint(500, 600)
        credit_tier = "Very Poor"
        annual_income = random.randint(50000, 70000)  # Medium income
        income_tier = "Medium"
        debt_ratio = random.uniform(0.3, 0.4)  # Medium debt
        debt_tier = "Medium"
        payment_history = "Fair"
        employment_status = "Employed"
    
    elif approval_type == "decline_low_income":
        # This application should be declined due to low income
        credit_score = random.randint(680, 720)  # Good but not excellent
        credit_tier = "Good"
        annual_income = random.randint(20000, 35000)  # Low income
        income_tier = "Low"
        debt_ratio = random.uniform(0.3, 0.4)  # Medium debt
        debt_tier = "Medium"
        payment_history = "Good"
        employment_status = "Part-time"
    
    elif approval_type == "decline_high_debt":
        # This application should be declined due to high debt
        credit_score = random.randint(680, 720)  # Good but not excellent
        credit_tier = "Good"
        annual_income = random.randint(50000, 70000)  # Medium income
        income_tier = "Medium"
        debt_ratio = random.uniform(0.7, 0.8)  # High debt
        debt_tier = "High"
        payment_history = "Good"
        employment_status = "Employed"
    
    else:
        # Random application with slight variation
        credit_segments = [
            (500, 600, "Very Poor"),
            (601, 680, "Poor"),
            (681, 740, "Good"),
            (741, 800, "Very Good"),
            (801, 850, "Excellent")
        ]
        segment = random.choice(credit_segments)
        credit_score = random.randint(segment[0], segment[1])
        credit_tier = segment[2]
        
        income_segments = [
            (20000, 40000, "Low"),
            (40001, 70000, "Medium"),
            (70001, 100000, "High"),
            (100001, 150000, "Very High")
        ]
        income_segment = random.choice(income_segments)
        annual_income = random.randint(income_segment[0], income_segment[1])
        income_tier = income_segment[2]
        
        debt_segments = [
            (0.1, 0.2, "Very Low"),
            (0.21, 0.4, "Low"),
            (0.41, 0.6, "Medium"),
            (0.61, 0.8, "High")
        ]
        debt_segment = random.choice(debt_segments)
        debt_ratio = random.uniform(debt_segment[0], debt_segment[1])
        debt_tier = debt_segment[2]
        
        payment_options = ["Excellent", "Good", "Fair", "Poor"]
        payment_history = random.choice(payment_options)
        
        employment_options = ["Employed", "Self-employed", "Part-time", "Unemployed"]
        employment_status = random.choice(employment_options)
    
    existing_debt = int(annual_income * debt_ratio)
    
    return {
        "personalDetails": {
            "name": f"Applicant {random.randint(1000, 9999)}",
            "age": random.randint(21, 65),
            "address": "123 Main St",
            "phoneNumber": "555-1234"
        },
        "creditHistory": {
            "creditScore": credit_score,
            "creditTier": credit_tier,
            "paymentHistory": payment_history,
            "creditUtilization": random.randint(10, 90)
        },
        "financialInformation": {
            "annualIncome": annual_income,
            "incomeTier": income_tier,
            "existingDebt": existing_debt,
            "debtRatio": debt_ratio,
            "debtTier": debt_tier,
            "employmentStatus": employment_status,
            "monthlyExpenses": random.randint(1000, 5000)
        }
    }

def should_approve(app):
    """Simple and clear approval criteria that's easy to rediscover"""
    # HIGH CREDIT SCORE - Excellent credit always gets approved
    if app["creditHistory"]["creditTier"] == "Excellent":
        return True
    
    # HIGH INCOME - Very high income with good credit gets approved
    if app["financialInformation"]["incomeTier"] == "Very High" and app["creditHistory"]["creditTier"] in ["Good", "Very Good"]:
        return True
    
    # LOW DEBT - Very low debt with good credit gets approved
    if app["financialInformation"]["debtTier"] == "Very Low" and app["creditHistory"]["creditTier"] in ["Good", "Very Good"]:
        return True
    
    # Decline if not satisfying any of above rules
    return False

def generate_new_applications():
    """Generate 20 applications with crystal-clear approval patterns"""
    application_count = 20
    
    # Clear previous files
    for existing_file in os.listdir(APPLICATIONS_DIR):
        file_path = os.path.join(APPLICATIONS_DIR, existing_file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # Clear previous results
    for existing_file in os.listdir(RESULTS_DIR):
        file_path = os.path.join(RESULTS_DIR, existing_file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # Define application types to generate
    application_types = [
        "approve_high_score", "approve_high_score",  # High credit score approvals
        "approve_high_income", "approve_high_income",  # High income approvals
        "approve_low_debt", "approve_low_debt",      # Low debt approvals
        "decline_low_score", "decline_low_score",    # Low score declines
        "decline_low_income", "decline_low_income",  # Low income declines
        "decline_high_debt", "decline_high_debt",    # High debt declines
        None, None, None, None, None, None, None, None  # Random applications
    ]
    
    applications = []
    approval_decisions = {}
    
    # Generate applications with clear patterns
    for i in range(1, application_count + 1):
        app_type = application_types[i-1]
        application = generate_application(app_type)
        is_approved = should_approve(application)
        
        applications.append(application)
        approval_decisions[str(i)] = is_approved
        
        # Save application to file
        with open(os.path.join(APPLICATIONS_DIR, f"application_{i}.json"), 'w') as f:
            json.dump(application, f, indent=2)
    
    # Save hidden approvals
    with open(os.path.join(APPLICATIONS_DIR, "hidden_approvals.json"), 'w') as f:
        json.dump(approval_decisions, f, indent=2)
    
    print(f"Generated {len(applications)} applications with clear patterns")
    approved_count = sum(1 for v in approval_decisions.values() if v)
    print(f"Approval rate: {approved_count}/{len(applications)} ({approved_count/len(applications)*100:.1f}%)")
    
    # Print examples to show patterns
    print("\nExample applications (showing key factors):")
    for i in range(1, len(applications) + 1):
        app = applications[i-1]
        approved = approval_decisions[str(i)]
        credit_tier = app["creditHistory"]["creditTier"]
        income_tier = app["financialInformation"]["incomeTier"]
        debt_tier = app["financialInformation"]["debtTier"]
        
        print(f"App {i}: Credit={credit_tier}, Income={income_tier}, Debt={debt_tier} → {'APPROVED' if approved else 'DECLINED'}")

if __name__ == "__main__":
    generate_new_applications() 