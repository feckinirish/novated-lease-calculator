import os
import math
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
# It's best practice to set FLASK_SECRET_KEY in your .env file
# e.g., FLASK_SECRET_KEY=your_very_long_and_random_secret_key_here
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'a_fallback_secret_key_for_dev')

# Configure Google Generative AI with your API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please set it.")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Generative Model (using Gemini 1.5 Flash for speed)
# 'gemini-1.5-flash-latest' is generally good for conversational tasks.
model = genai.GenerativeModel('gemini-1.5-flash-latest')


# --- Constants ---
# From 2024-2025 Australian tax rates
# 0 - 18,200: 0%
# 18,201 - 45,000: 19% (Tax on $18,200 is $0)
# 45,001 - 135,000: 32.5% (Tax on $45,000 is $5,092)
# 135,001 - 190,000: 37% (Tax on $135,000 is $34,342)
# 190,001+: 45% (Tax on $190,000 is $54,692)

TAX_BRACKETS_CUMULATIVE = [
    (0, 0.0, 0),         # Up to $18,200
    (18200, 0.19, 0),    # $18,201 to $45,000
    (45000, 0.325, 5092), # $45,001 to $135,000
    (135000, 0.37, 34342),# $135,000 to $190,000
    (190000, 0.45, 54692) # $190,000 and over
]

# --- Backend Calculation Functions ---
# These functions should return NUMERICAL values for further calculations

def calculate_income_tax(gross_salary):
    tax = 0
    for i in range(len(TAX_BRACKETS_CUMULATIVE)):
        threshold, rate, cumulative_tax = TAX_BRACKETS_CUMULATIVE[i]
        if gross_salary > threshold:
            if i == len(TAX_BRACKETS_CUMULATIVE) - 1:
                tax = cumulative_tax + (gross_salary - threshold) * rate
                break
            next_threshold = TAX_BRACKETS_CUMULATIVE[i+1][0]
            if gross_salary <= next_threshold:
                tax = cumulative_tax + (gross_salary - threshold) * rate
                break
        else:
            break
    return max(0, tax)

def calculate_medicare_levy(gross_salary):
    return gross_salary * 0.02

def calculate_mls(gross_salary, private_health_cover=False):
    if private_health_cover:
        return 0
    # Tier 1, 2, 3 thresholds for MLS
    elif gross_salary > 140000: # Tier 3
        return gross_salary * 0.015
    elif gross_salary > 105000: # Tier 2
        return gross_salary * 0.0125
    elif gross_salary > 93000: # Tier 1
        return gross_salary * 0.01
    return 0

def calculate_total_tax_liability(gross_salary, private_health_cover=False):
    income_tax = calculate_income_tax(gross_salary)
    medicare_levy = calculate_medicare_levy(gross_salary)
    mls = calculate_mls(gross_salary, private_health_cover)
    return income_tax + medicare_levy + mls

def calculate_gst_on_vehicle(vehicle_price):
    return vehicle_price / 11

def calculate_gst_on_running_costs(annual_running_costs):
    return annual_running_costs / 11

def calculate_fbt(vehicle_price, annual_kms, is_ev_phv, statutory_formula_method=True):
    # FBT Exempt for qualifying EVs/PHEVs in Australia
    if is_ev_phv:
        return 0
    # Statutory formula method for non-EVs
    # Assumes a 20% statutory rate, irrespective of kms travelled
    statutory_base_value = vehicle_price * 0.20
    return statutory_base_value

def calculate_lease_finance_payment(vehicle_price, residual_value, lease_term_years, lease_interest_rate_annual):
    if lease_term_years <= 0:
        return 0

    monthly_rate = lease_interest_rate_annual / 12 # Already converted from percentage in index()
    num_payments = lease_term_years * 12

    try:
        if monthly_rate == 0:
            if num_payments == 0: return 0
            # Simple division if no interest
            amortized_payment_monthly = (vehicle_price - residual_value) / num_payments
            total_monthly_payment = amortized_payment_monthly
        else:
            # Annuity formula for principal amortization (PMT = [ i * PV ] / [ 1 - ( 1 + i )^-n ])
            amortized_principal = vehicle_price - residual_value
            # Handle residual interest component
            # This is a simplified PMT for principal only, then adding residual interest.
            # A true lease PMT calculation would be more complex and usually done by finance providers.
            # This aims to approximate based on the logic seen in some basic calculators.
            amortized_payment_monthly = (monthly_rate * amortized_principal) / (1 - math.pow(1 + monthly_rate, -num_payments))
            residual_interest_monthly = (residual_value * monthly_rate) # Simple interest on residual

            total_monthly_payment = amortized_payment_monthly + residual_interest_monthly
        return total_monthly_payment * 12 # Return annual payment

    except Exception as e:
        print(f"Error in calculate_lease_finance_payment: {e}")
        # Fallback for division by zero or other math errors if term is 0 or rates are extreme
        if lease_term_years == 0: return 0
        return (vehicle_price - residual_value) / lease_term_years * 12 # Simple linear if error


def calculate_novated_lease(gross_salary, vehicle_price, lease_term_years, annual_kms, is_ev_phv,
                             lease_interest_rate_annual, total_annual_running_costs, private_health_cover=False):

    annual_fbt = calculate_fbt(vehicle_price, annual_kms, is_ev_phv)

    # ATO specified residual percentages for different lease terms
    residual_percentages = {1: 0.6563, 2: 0.5625, 3: 0.4688, 4: 0.3750, 5: 0.2813}
    residual_percentage = residual_percentages.get(lease_term_years, 0.4688) # Default to 3-year if unknown
    residual_value = vehicle_price * residual_percentage

    annual_lease_finance_payment = calculate_lease_finance_payment(
        vehicle_price, residual_value, lease_term_years, lease_interest_rate_annual
    )

    annual_gst_saving_running_costs = calculate_gst_on_running_costs(total_annual_running_costs)
    one_off_gst_saving_vehicle = calculate_gst_on_vehicle(vehicle_price)

    # Pre-tax salary sacrifice includes lease payments and running costs
    annual_pre_tax_salary_sacrifice = annual_lease_finance_payment + total_annual_running_costs
    # FBT is typically paid by employee via post-tax contributions (EOY payment or spread across year)
    annual_fbt_paid_by_employee_post_tax = annual_fbt

    # Calculate tax savings
    taxable_income_after_sacrifice = gross_salary - annual_pre_tax_salary_sacrifice
    # Ensure taxable income doesn't go negative for tax calculation purposes
    if taxable_income_after_sacrifice < 0:
        taxable_income_after_sacrifice = 0

    original_tax_liability = calculate_total_tax_liability(gross_salary, private_health_cover)
    new_tax_liability = calculate_total_tax_liability(taxable_income_after_sacrifice, private_health_cover)
    estimated_annual_income_tax_saving = original_tax_liability - new_tax_liability

    # Total Net Cost Over Term for Novated Lease
    # This is the "true" cost considering all benefits/liabilities over the term
    total_net_cost_over_term = (annual_lease_finance_payment * lease_term_years) + \
                               (total_annual_running_costs * lease_term_years) + \
                               (annual_fbt_paid_by_employee_post_tax * lease_term_years) + \
                               residual_value - \
                               (estimated_annual_income_tax_saving * lease_term_years) - \
                               (annual_gst_saving_running_costs * lease_term_years) - \
                               one_off_gst_saving_vehicle

    effective_annual_cost = total_net_cost_over_term / lease_term_years if lease_term_years > 0 else total_net_cost_over_term
    effective_monthly_cost = effective_annual_cost / 12 if effective_annual_cost > 0 else 0

    return {
        "Annual Lease Finance Payment": annual_lease_finance_payment,
        "Annual Running Costs Packaged": total_annual_running_costs, # This is the packaged amount
        "Annual Fbt Paid By Employee Post Tax": annual_fbt_paid_by_employee_post_tax,
        "Annual Pre Tax Salary Sacrifice": annual_pre_tax_salary_sacrifice,
        "Estimated Annual Income Tax Saving": estimated_annual_income_tax_saving,
        "Annual Gst Saving On Running Costs": annual_gst_saving_running_costs,
        "One Off Gst Saving On Vehicle": one_off_gst_saving_vehicle,
        "Residual Value": residual_value,
        "Total Net Cost Over Term": total_net_cost_over_term, # NUMERICAL
        "Effective Monthly Cost": effective_monthly_cost,
        "Effective Annual Cost": effective_annual_cost
    }

def calculate_car_loan(vehicle_price, loan_term_years, loan_interest_rate_annual):
    if loan_term_years <= 0:
        return {
            "Loan Principal": vehicle_price,
            "Annual Repayment Approx": 0,
            "Total Interest Paid Over Term": 0,
            "Total Repayments Over Term": vehicle_price,
        }

    monthly_rate = loan_interest_rate_annual / 12 # Already converted from percentage in index()
    num_payments = loan_term_years * 12

    if monthly_rate == 0:
        monthly_repayment = vehicle_price / num_payments
    else:
        # Standard loan repayment (PMT) formula
        monthly_repayment = (monthly_rate * vehicle_price) / (1 - math.pow(1 + monthly_rate, -num_payments))

    annual_repayment_approx = monthly_repayment * 12
    total_repayments_over_term = monthly_repayment * num_payments
    total_interest_paid_over_term = total_repayments_over_term - vehicle_price

    return {
        "Loan Principal": vehicle_price,
        "Annual Repayment Approx": annual_repayment_approx,
        "Total Interest Paid Over Term": total_interest_paid_over_term,
        "Total Repayments Over Term": total_repayments_over_term, # NUMERICAL
    }

def calculate_outright_purchase(vehicle_price): # loan_term_years is not needed here
    # Outright purchase simply involves the initial outlay
    return {
        "Initial Outlay": vehicle_price,
        "Total Cost Over Term": vehicle_price, # NUMERICAL (before running costs added in index)
    }

# --- Flask Routes ---

# The format_currency function is handled in JS now for inputs
# and directly in the Python function for results before passing to template.

@app.route('/', methods=['GET', 'POST'])
def index():
    # Default values for inputs - these are numerical and used for initial display
    # and fallback in case of errors.
    defaults = {
        'salary': 120000,
        'phc': 'N',
        'vehicle_price': 45000,
        'lease_term': 3,
        'annual_kms': 15000,
        'is_ev': 'N',
        'nl_interest_rate': 7.0, # These are percentages
        'loan_interest_rate': 8.5, # These are percentages
        'fuel_cost': 2000,
        'service_tyres': 500,
        'insurance': 1000,
        'registration': 800
    }
    results = None # Initialize results to None

    # This dictionary will hold the inputs as they are passed to the template (either defaults or from form)
    current_inputs = defaults.copy()

    if request.method == 'POST':
        try:
            # --- Input Parsing and Validation ---
            # Try to get values from form, fall back to defaults if not present
            # or if conversion fails (though frontend validation should ideally prevent this for sliders)
            gross_salary = float(request.form.get('salary', defaults['salary']))
            private_health_cover = True if request.form.get('phc', defaults['phc']).upper() == 'Y' else False
            vehicle_price = float(request.form.get('vehicle_price', defaults['vehicle_price']))
            lease_term_years = int(request.form.get('lease_term', defaults['lease_term']))
            annual_kms = float(request.form.get('annual_kms', defaults['annual_kms']))
            is_ev_phv = True if request.form.get('is_ev', defaults['is_ev']).upper() == 'Y' else False
            novated_lease_interest_rate_percent = float(request.form.get('nl_interest_rate', defaults['nl_interest_rate']))
            car_loan_interest_rate_percent = float(request.form.get('loan_interest_rate', defaults['loan_interest_rate']))
            running_costs_fuel = float(request.form.get('fuel_cost', defaults['fuel_cost']))
            running_costs_service_tyres = float(request.form.get('service_tyres', defaults['service_tyres']))
            running_costs_insurance = float(request.form.get('insurance', defaults['insurance']))
            running_costs_registration = float(request.form.get('registration', defaults['registration']))

            # Update current_inputs to reflect what was submitted for sticky form fields
            current_inputs.update({
                'salary': gross_salary,
                'phc': request.form.get('phc', defaults['phc']),
                'vehicle_price': vehicle_price,
                'lease_term': lease_term_years,
                'annual_kms': annual_kms,
                'is_ev': request.form.get('is_ev', defaults['is_ev']),
                'nl_interest_rate': novated_lease_interest_rate_percent,
                'loan_interest_rate': car_loan_interest_rate_percent,
                'fuel_cost': running_costs_fuel,
                'service_tyres': running_costs_service_tyres,
                'insurance': running_costs_insurance,
                'registration': running_costs_registration
            })

            # Input Validation
            MIN_INTEREST_RATE_THRESHOLD = 0.0001
            errors = []
            if gross_salary <= 0: errors.append("Gross Annual Salary must be a positive number.")
            if vehicle_price <= 0: errors.append("Vehicle Price must be a positive number.")
            if lease_term_years <= 0: errors.append("Lease Term (Years) must be a positive integer.")
            if annual_kms < 0: errors.append("Annual Kilometers cannot be negative.")
            if running_costs_fuel < 0 or running_costs_service_tyres < 0 or \
               running_costs_insurance < 0 or running_costs_registration < 0:
                errors.append("Running costs cannot be negative.")
            if (novated_lease_interest_rate_percent < 0) or \
               (novated_lease_interest_rate_percent > 0 and novated_lease_interest_rate_percent < MIN_INTEREST_RATE_THRESHOLD):
                 errors.append(f"Novated Lease Interest Rate must be 0 or at least {MIN_INTEREST_RATE_THRESHOLD:.4f}% (and cannot be negative).")
            if (car_loan_interest_rate_percent < 0) or \
               (car_loan_interest_rate_percent > 0 and car_loan_interest_rate_percent < MIN_INTEREST_RATE_THRESHOLD):
                 errors.append(f"Car Loan Interest Rate must be 0 or at least {MIN_INTEREST_RATE_THRESHOLD:.4f}% (and cannot be negative).")

            if errors:
                for error in errors:
                    flash(error, 'error')
                return render_template('index.html', results=results, inputs=current_inputs)

            # Convert percentage rates to decimals for calculations
            novated_lease_interest_rate = novated_lease_interest_rate_percent / 100
            car_loan_interest_rate = car_loan_interest_rate_percent / 100
            total_annual_running_costs = running_costs_fuel + running_costs_service_tyres + running_costs_insurance + running_costs_registration
            monthly_running_costs = total_annual_running_costs / 12

            # --- Perform Calculations (NUMERICAL RESULTS FIRST) ---
            nl_calc_raw = calculate_novated_lease(
                gross_salary, vehicle_price, lease_term_years, annual_kms, is_ev_phv,
                novated_lease_interest_rate, total_annual_running_costs, private_health_cover
            )

             # --- ADD THIS PRINT STATEMENT ---
            print("\n--- Raw Novated Lease Calculation Results (nl_calc_raw) ---")
            for key, value in nl_calc_raw.items():
                print(f"{key}: {value}")
            print("----------------------------------------------------\n")           

            loan_calc_raw = calculate_car_loan(
                vehicle_price, lease_term_years, car_loan_interest_rate
            )

            outright_calc_raw = calculate_outright_purchase(vehicle_price)

            # Adjust loan and outright total costs to include running costs for comparison
            loan_total_cost_over_term_numeric = loan_calc_raw["Total Repayments Over Term"] + \
                                                 (total_annual_running_costs * lease_term_years)
            outright_total_cost_over_term_numeric = outright_calc_raw["Initial Outlay"] + \
                                                     (total_annual_running_costs * lease_term_years)

            # --- Build the results dictionary for display (FORMATTED STRINGS) ---
            results = {
                'nl': {
                    'Annual Lease Finance Payment': '{:,.2f}'.format(nl_calc_raw["Annual Lease Finance Payment"]),
                    'Annual Running Costs Packaged': '{:,.2f}'.format(nl_calc_raw["Annual Running Costs Packaged"]),
                    'Annual FBT Paid By Employee Post Tax': '{:,.2f}'.format(nl_calc_raw["Annual Fbt Paid By Employee Post Tax"]),
                    'Annual Pre Tax Salary Sacrifice': '{:,.2f}'.format(nl_calc_raw["Annual Pre Tax Salary Sacrifice"]),
                    'Estimated Annual Income Tax Saving': '{:,.2f}'.format(nl_calc_raw["Estimated Annual Income Tax Saving"]),
                    'Annual GST Saving On Running Costs': '{:,.2f}'.format(nl_calc_raw["Annual Gst Saving On Running Costs"]),
                    'One Off GST Saving On Vehicle': '{:,.2f}'.format(nl_calc_raw["One Off Gst Saving On Vehicle"]),
                    'Residual Value': '{:,.2f}'.format(nl_calc_raw["Residual Value"]),
                    'Total Net Cost Over Term': '{:,.2f}'.format(nl_calc_raw["Total Net Cost Over Term"]),
                    'Effective Monthly Cost': '{:,.2f}'.format(nl_calc_raw["Effective Monthly Cost"]),
                    'Effective Annual Cost': '{:,.2f}'.format(nl_calc_raw["Effective Annual Cost"])
                },
                'loan': {
                    'Annual Repayment Approx': '{:,.2f}'.format(loan_calc_raw["Annual Repayment Approx"]),
                    'Total Interest Paid Over Term': '{:,.2f}'.format(loan_calc_raw["Total Interest Paid Over Term"]),
                    'Total Repayments Over Term': '{:,.2f}'.format(loan_calc_raw["Total Repayments Over Term"]), # Loan repayments only
                    'Monthly Running Costs': '{:,.2f}'.format(monthly_running_costs),
                    'Annual Running Costs': '{:,.2f}'.format(total_annual_running_costs),
                    'Effective Monthly Cost (Loan only)': '{:,.2f}'.format(loan_calc_raw["Annual Repayment Approx"] / 12),
                    'Monthly Loan Cost (incl. running)': '{:,.2f}'.format(loan_total_cost_over_term_numeric / (lease_term_years * 12)),
                    'Total Cost Over Term (Loan incl. running)': '{:,.2f}'.format(loan_total_cost_over_term_numeric) # NUMERICAL for display
                },
                'outright': {
                    'Initial Outlay': '{:,.2f}'.format(outright_calc_raw["Initial Outlay"]),
                    'Total Cost Over Term': '{:,.2f}'.format(outright_total_cost_over_term_numeric), # NUMERICAL for display
                    'Effective Monthly Cost': '{:,.2f}'.format(outright_total_cost_over_term_numeric / (lease_term_years * 12))
                },
                'comparison': {
                    # THESE ARE NOW CALCULATED DYNAMICALLY using the numerical totals
                    'savings_vs_loan': '{:,.2f}'.format(loan_total_cost_over_term_numeric - nl_calc_raw["Total Net Cost Over Term"]),
                    'savings_vs_outright': '{:,.2f}'.format(outright_total_cost_over_term_numeric - nl_calc_raw["Total Net Cost Over Term"])
                }
            }

            flash("Calculation successful!", "success")

        except ValueError as e:
            flash(f"Invalid input: {e}. Please ensure all monetary and numerical fields contain valid numbers.", 'error')
            # current_inputs is already updated by request.form.get, or defaults
        except Exception as e:
            flash(f"An unexpected error occurred during calculation: {e}", 'error')
            # current_inputs is already updated by request.form.get, or defaults

    # For GET requests (initial page load) or if a POST request had errors
    # current_inputs already contains either defaults or values from last form submission
    return render_template('index.html', results=results, inputs=current_inputs)


@app.route('/faq')
def faq():
    return render_template('faq.html')


# --- AI Chat Routes ---

@app.route('/chat')
def chat_page():
    """Renders the main chat interface page (standalone chat)."""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Handles chat messages, sends to AI, and returns AI response."""
    data = request.json
    user_message = data.get('message')
    results = data.get('results') # Get the results data from the frontend
    inputs = data.get('inputs')   # Get the inputs data from the frontend

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # --- Prompt Engineering: Constructing the context for the AI ---
    # This prompt provides the AI with the context of the calculator inputs and results.
    context_message = """You are an AI assistant specialized in Australian novated leases and car finance.
    The user is interacting with a Novated Lease & Car Comparison Calculator.
    All financial figures mentioned by the user and in the provided data are in AUD.

    Here are the current calculator inputs:
    """
    if inputs:
        for key, value in inputs.items():
            context_message += f"- {key.replace('_', ' ').title()}: {value}\n"
    else:
        context_message += "No specific inputs were provided.\n"

    context_message += """
    Here are the current calculation results:
    """
    if results:
        # Iterate over categories (nl, loan, outright, comparison)
        for category, category_results in results.items():
            context_message += f"\n**{category.replace('_', ' ').title()} Results:**\n"
            # Iterate over each label-value pair within the category
            for label, value in category_results.items():
                context_message += f"- {label}: ${value}\n"
    else:
        context_message += "No calculation results available.\n"

    context_message += """

    **Instructions for your response:**
    1.  **Directly answer the user's question** based on the provided inputs and calculation results.
    2.  If the user asks for the **definition or meaning of a specific term** (e.g., 'Residual Value', 'FBT', 'Salary Sacrifice', 'Novated Lease', 'Effective Monthly Cost', 'GST Savings') that is present in the provided context, **define that term clearly and concisely, directly referencing the user's results if applicable.**
    3.  If the user asks about a specific numerical value from the results, state the number clearly.
    4.  If the question is general and cannot be answered directly from the provided results, give a general explanation relevant to Australian novated leases or car finance.
    5.  Be concise, helpful, and professional.
    6.  **Do not make up any numbers or details not explicitly provided** in the `User Inputs` or `Calculation Results` sections.
    7.  Always state that all financial figures you mention are in AUD.
    """

    full_prompt = context_message + "User's question: " + user_message

    try:
        response = model.generate_content(full_prompt)
        ai_response = response.text

        return jsonify({'response': ai_response})

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({'error': 'Failed to get response from AI'}), 500

if __name__ == '__main__':
    # Add a note about debug mode for clarity during development
    # In a production environment, debug=False would be used.
    app.run(debug=True)