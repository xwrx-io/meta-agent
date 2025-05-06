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
    """Generate application with nuanced patterns based on approval_type"""
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
    
    elif approval_type == "approve_balanced_factors":
        # This application should be approved due to balanced positive factors
        credit_score = random.randint(720, 760)  # Very good credit
        credit_tier = "Very Good"
        annual_income = random.randint(75000, 95000)  # High income
        income_tier = "High"
        debt_ratio = random.uniform(0.2, 0.3)  # Low debt
        debt_tier = "Low"
        payment_history = "Excellent"
        employment_status = "Employed"
    
    elif approval_type == "approve_high_income_high_debt":
        # This application should be approved due to very high income despite high debt
        credit_score = random.randint(680, 720)  # Good credit
        credit_tier = "Good"
        annual_income = random.randint(120000, 180000)  # Very high income
        income_tier = "Very High"
        debt_ratio = random.uniform(0.5, 0.6)  # Medium-high debt
        debt_tier = "Medium"
        payment_history = "Good"
        employment_status = "Employed"
    
    elif approval_type == "approve_edge_case":
        # Edge case approval - medium everything but long payment history
        credit_score = random.randint(650, 680)  # Medium credit score
        credit_tier = "Good"
        annual_income = random.randint(60000, 75000)  # Medium-high income
        income_tier = "Medium"
        debt_ratio = random.uniform(0.3, 0.4)  # Medium debt
        debt_tier = "Medium"
        payment_history = "Excellent"  # Strong payment history
        employment_status = "Self-employed"
    
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
    
    elif approval_type == "decline_multiple_issues":
        # This application should be declined due to multiple issues
        credit_score = random.randint(620, 650)  # Poor to medium credit
        credit_tier = "Poor"
        annual_income = random.randint(35000, 45000)  # Low-medium income
        income_tier = "Low"
        debt_ratio = random.uniform(0.5, 0.6)  # Medium-high debt
        debt_tier = "Medium"
        payment_history = "Fair"
        employment_status = "Part-time"
    
    elif approval_type == "decline_edge_case":
        # Edge case decline - good credit but unemployed
        credit_score = random.randint(700, 750)  # Good credit score
        credit_tier = "Good"
        annual_income = random.randint(30000, 40000)  # Low income
        income_tier = "Low"
        debt_ratio = random.uniform(0.4, 0.5)  # Medium debt
        debt_tier = "Medium"
        payment_history = "Good"
        employment_status = "Unemployed"
    
    else:
        # Random application with slight variation
        credit_segments = [
            (500, 600, "Very Poor"),
            (601, 650, "Poor"),
            (651, 700, "Good"),
            (701, 770, "Very Good"),
            (771, 850, "Excellent")
        ]
        segment = random.choice(credit_segments)
        credit_score = random.randint(segment[0], segment[1])
        credit_tier = segment[2]
        
        income_segments = [
            (20000, 40000, "Low"),
            (40001, 70000, "Medium"),
            (70001, 100000, "High"),
            (100001, 180000, "Very High")
        ]
        income_segment = random.choice(income_segments)
        annual_income = random.randint(income_segment[0], income_segment[1])
        income_tier = income_segment[2]
        
        debt_segments = [
            (0.1, 0.2, "Very Low"),
            (0.21, 0.35, "Low"),
            (0.36, 0.6, "Medium"),
            (0.61, 0.8, "High")
        ]
        debt_segment = random.choice(debt_segments)
        debt_ratio = random.uniform(debt_segment[0], debt_segment[1])
        debt_tier = debt_segment[2]
        
        payment_options = ["Excellent", "Good", "Fair", "Poor"]
        payment_history = random.choice(payment_options)
        
        employment_options = ["Employed", "Self-employed", "Part-time", "Unemployed", "Contract"]
        employment_status = random.choice(employment_options)
    
    existing_debt = int(annual_income * debt_ratio)
    
    # Add a credit age/history length for more nuance
    credit_age = random.randint(1, 25)
    
    # Add previous applications and credit inquiries count
    inquiries = random.randint(0, 10)
    
    # Add monthly income for more granular assessment
    monthly_income = int(annual_income / 12)
    
    # Add savings data
    savings = random.randint(int(annual_income * 0.1), int(annual_income * 1.2))
    
    return {
        "personalDetails": {
            "name": f"Applicant {random.randint(1000, 9999)}",
            "age": random.randint(21, 75),
            "address": f"{random.randint(100, 999)} Main St",
            "phoneNumber": f"555-{random.randint(1000, 9999)}",
            "yearsAtAddress": random.randint(1, 20)
        },
        "creditHistory": {
            "creditScore": credit_score,
            "creditTier": credit_tier,
            "paymentHistory": payment_history,
            "creditUtilization": random.randint(10, 90),
            "creditAgeYears": credit_age,
            "recentInquiries": inquiries
        },
        "financialInformation": {
            "annualIncome": annual_income,
            "monthlyIncome": monthly_income,
            "incomeTier": income_tier,
            "existingDebt": existing_debt,
            "debtRatio": debt_ratio,
            "debtTier": debt_tier,
            "employmentStatus": employment_status,
            "yearsEmployed": random.randint(1, 15),
            "monthlyExpenses": random.randint(1000, 5000),
            "savings": savings
        }
    }

def should_approve(app):
    """More nuanced approval criteria with subtle patterns"""
    
    # RULE 1: Excellent credit always gets approved
    if app["creditHistory"]["creditTier"] == "Excellent":
        return True
    
    # RULE 2: Good credit with very low debt gets approved
    if app["creditHistory"]["creditTier"] in ["Good", "Very Good"] and app["financialInformation"]["debtTier"] == "Very Low":
        return True
    
    # RULE 3: High/Very High income with Low/Medium debt gets approved if credit is at least Good
    if (app["financialInformation"]["incomeTier"] in ["High", "Very High"] and 
        app["financialInformation"]["debtTier"] in ["Low", "Medium"] and
        app["creditHistory"]["creditTier"] in ["Good", "Very Good", "Excellent"]):
        return True
    
    # RULE 4: Good payment history can overcome slightly lower credit score if other factors are strong
    if (app["creditHistory"]["paymentHistory"] == "Excellent" and
        app["creditHistory"]["creditTier"] in ["Good"] and
        app["financialInformation"]["debtTier"] in ["Low", "Very Low"] and
        app["financialInformation"]["incomeTier"] in ["Medium", "High", "Very High"]):
        return True
    
    # RULE 5: Unemployed applicants are rejected regardless of other factors
    if app["financialInformation"]["employmentStatus"] == "Unemployed":
        return False
    
    # RULE 6: Very Poor credit gets rejected regardless of other factors
    if app["creditHistory"]["creditTier"] == "Very Poor":
        return False
    
    # Add a few exceptions (noise) to make the pattern more challenging to discover
    # This creates a small level of inconsistency that the ML needs to work through
    
    # 10% chance to approve a borderline case that would normally be rejected
    if (app["creditHistory"]["creditTier"] == "Good" and
        random.random() < 0.1 and  # 10% chance
        app["financialInformation"]["incomeTier"] in ["Medium"] and
        app["financialInformation"]["debtTier"] != "High"):
        return True
    
    # 10% chance to reject a borderline case that would normally be approved
    if (app["creditHistory"]["creditTier"] == "Good" and
        random.random() < 0.1 and  # 10% chance
        app["financialInformation"]["incomeTier"] in ["High"] and
        app["financialInformation"]["debtTier"] in ["Medium"]):
        return False
    
    # Decline if not satisfying any of above rules
    return False

def generate_new_applications():
    """Generate 50 applications with nuanced approval patterns"""
    application_count = 50
    
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
    
    # Define application types to generate with more variety
    application_types = [
        # Standard approve cases
        "approve_high_score", "approve_high_score", "approve_high_score",
        "approve_high_income", "approve_high_income", "approve_high_income",
        "approve_low_debt", "approve_low_debt", "approve_low_debt",
        
        # New approve cases
        "approve_balanced_factors", "approve_balanced_factors",
        "approve_high_income_high_debt", "approve_high_income_high_debt",
        "approve_edge_case", "approve_edge_case",
        
        # Standard decline cases
        "decline_low_score", "decline_low_score", "decline_low_score",
        "decline_low_income", "decline_low_income", "decline_low_income",
        "decline_high_debt", "decline_high_debt", "decline_high_debt",
        
        # New decline cases
        "decline_multiple_issues", "decline_multiple_issues",
        "decline_edge_case", "decline_edge_case",
    ]
    
    # Fill the rest with random applications
    while len(application_types) < application_count:
        application_types.append(None)
    
    # Shuffle to randomize order
    random.shuffle(application_types)
    
    applications = []
    approval_decisions = {}
    
    # Generate applications with nuanced patterns
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
    
    print(f"Generated {len(applications)} applications with nuanced patterns")
    approved_count = sum(1 for v in approval_decisions.values() if v)
    print(f"Approval rate: {approved_count}/{len(applications)} ({approved_count/len(applications)*100:.1f}%)")
    
    # Print examples to show patterns
    print("\nExample applications (showing key factors):")
    for i in range(1, min(10, len(applications)) + 1):  # Show first 10 examples
        app = applications[i-1]
        approved = approval_decisions[str(i)]
        credit_tier = app["creditHistory"]["creditTier"]
        income_tier = app["financialInformation"]["incomeTier"]
        debt_tier = app["financialInformation"]["debtTier"]
        payment_history = app["creditHistory"]["paymentHistory"]
        employment = app["financialInformation"]["employmentStatus"]
        
        print(f"App {i}: Credit={credit_tier}, Income={income_tier}, Debt={debt_tier}, " +
              f"Payment={payment_history}, Employment={employment} → {'APPROVED' if approved else 'DECLINED'}")

if __name__ == "__main__":
    generate_new_applications() 