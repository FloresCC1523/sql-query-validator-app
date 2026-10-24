# app.py - Credit Risk Query Engine (Streamlit)
import os
import re
import json
import glob
import sqlite3
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Credit Risk Query Engine", page_icon="📊", layout="wide")

# ---------------------------------------------------------------
# Credentials: Streamlit secrets (deployed) or config.json (local)
# ---------------------------------------------------------------
def load_credentials():
    key, base = None, None
    try:
        key = st.secrets["OPENAI_API_KEY"]
        base = st.secrets.get("OPENAI_API_BASE")
    except Exception:
        pass
    if not key and os.path.exists("config.json"):
        with open("config.json", "r") as f:
            cfg = json.load(f)
        key = cfg.get("OPENAI_API_KEY")
        base = cfg.get("OPENAI_API_BASE")
    return key, base

OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Add OPENAI_API_KEY (and OPENAI_API_BASE) in the app's Secrets.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE

MODEL_NAME = "gpt-4o-mini"   # change if your course uses a different model

@st.cache_resource
def get_llms(api_key, api_base):
    # cached per key: changing the key in Secrets creates fresh models (no reboot needed)
    kwargs = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    return (ChatOpenAI(model=MODEL_NAME, temperature=0, **kwargs),
            ChatOpenAI(model=MODEL_NAME, temperature=0, **kwargs))

llm, evaluator_llm = get_llms(OPENAI_API_KEY, OPENAI_API_BASE)

# ---------------------------------------------------------------
# Database: finds the .db file next to app.py, opens it read-only
# ---------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def find_db():
    db_files = sorted(glob.glob(os.path.join(APP_DIR, "**", "*.db"), recursive=True))
    return db_files[0] if db_files else None

@st.cache_resource
def get_connection(db_path):
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)

db_path = find_db()
conn = get_connection(db_path) if db_path else None
db_name = os.path.basename(db_path) if db_path else None
if conn is None:
    st.error("No .db file found. Upload your SQLite database file to the repository next to app.py.")
    st.stop()

# ---------------------------------------------------------------
# Database schema (given to the LLM)
# ---------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# Verified query templates
sql_1 = """
SELECT
    s.sector_name,
    ROUND(SUM(l.total_outstanding) / 1e6, 2) AS total_outstanding_mn,
    ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                   THEN l.total_outstanding ELSE 0 END) / 1e6, 2) AS npa_outstanding_mn
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_mn DESC
"""

sql_2 = """
SELECT
    loan_category,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_mn DESC
"""

sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount) / 1e6, 2) AS total_ead_mn,
    ROUND(SUM(ecl_amount) / 1e6, 2) AS total_ecl_mn
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage
"""

sql_4 = """
SELECT
    s.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_pct
FROM provisioning p
JOIN loan_master l ON p.loan_account_number = l.loan_account_number
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_pct DESC
"""

sql_5 = """
SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1e6, 2) AS total_outstanding_mn,
    l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
ORDER BY l.total_outstanding DESC
LIMIT 10
"""

sql_6 = """
SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_mn
FROM loan_master
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_outstanding_mn DESC
LIMIT 5
"""

sql_7 = """
SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1e6, 2) AS total_outstanding_mn,
    l.days_past_due,
    l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC
"""

sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1e6, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY dpd_bucket
ORDER BY MIN(days_past_due)
"""

sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""

sql_10 = """
SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1e6, 2) AS total_ecl_mn
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date
"""

verified_query_library = {
    'VQ1': {'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors', 'sql': sql_1},
    'VQ2': {'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)', 'sql': sql_2},
    'VQ3': {'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter', 'sql': sql_3},
    'VQ4': {'description': 'Average provision coverage ratio by sector for the latest reporting quarter', 'sql': sql_4},
    'VQ5': {'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level', 'sql': sql_5},
    'VQ6': {'description': 'Top 5 largest exposures aggregated at the business group level', 'sql': sql_6},
    'VQ7': {'description': 'All overdue loan accounts with their days past due and asset classification', 'sql': sql_7},
    'VQ8': {'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio', 'sql': sql_8},
    'VQ9': {'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle', 'sql': sql_9},
    'VQ10': {'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time', 'sql': sql_10}
}

# Pipeline tools
def classify_intent(user_question, query_library):
    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
You are an intent classifier for a bank's credit risk analytics query engine.
Your job is to decide whether the user's question can be answered by one of the
pre-approved VERIFIED query templates, or whether new SQL must be GENERATED.

Decision rules:
1. Choose "verified" only when a template's full result already contains everything needed
   to answer the question. A later step can pick out the relevant rows, so a template that
   returns ALL sectors / categories / stages still matches a question about ONE of them
   (e.g., "How much is in Real Estate and how much is NPA?" -> VQ1).
2. Choose "generated" when the question needs columns, filters, metrics, calculations or
   time periods that no template returns (for example: restructured loans, interest rates,
   impaired percentages, or one IFRS 9 stage tracked across several quarters).
3. Questions about how loans or the overdue book are DISTRIBUTED across DPD buckets or the
   aging profile -> VQ8. Requests to LIST individual overdue accounts -> VQ7.
4. If you are not sure a template fully answers the question, choose "generated".

User question:
{user_question}

Verified query templates:
{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context):
    generation_prompt = f"""
You are an expert SQL analyst for a bank's credit risk team. Write ONE SQLite query
that answers the user's question using only the database schema below.

Rules:
- Return a single read-only statement that starts with SELECT or WITH. No semicolon at the end.
- Use only tables and columns that exist in the schema. Join tables on the keys described.
- NPA, non-performing, and impaired loans mean asset_classification IN ('Substandard', 'Doubtful', 'Loss').
- "Latest" or "current" quarter means reporting_date = '2025-09-30' (provisioning)
  or rating_date = '2025-09-30' (borrower_rating).
- loan_master has no date column (it is a current snapshot): never add a date filter to it.
- For trends "over quarters", GROUP BY reporting_date and ORDER BY reporting_date (no single-date filter).
- Show money amounts in USD millions: ROUND(SUM(column) / 1e6, 2), with an alias ending in _mn.
- Calculate percentages with 100.0 * ... so they are not rounded to whole numbers, and ROUND to 2 decimals.
- If the question asks for a list AND an overall figure (such as a total percentage),
  return the list and include the overall figure as an extra column using a window function
  (e.g., SUM(...) OVER ()).
- Use clear column aliases and sort the result in the way the question asks.
- Return ONLY the SQL. No explanation, no markdown, no code fences.

User question:
{user_question}

Database schema:
{schema_context}
"""

    sql = llm.invoke(generation_prompt).content.strip()
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    result = {'passed': False, 'failed_check': None, 'details': '', 'relevance_confidence': None}

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select','from','where','and','or','group','by','order','having','limit','join','on','as','case',
                    'when','then','else','end','sum','count','avg','min','max','round','desc','asc','left','right',
                    'inner','outer','distinct','null','is','not','in','like','with','union','all','between','coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""
You are a senior credit risk data reviewer. Check whether the SQL query below correctly
answers the user's question. Review:
- Are the right tables and joins used?
- Are the right metrics and columns used (e.g., total_outstanding, ead_amount, ecl_amount, interest_rate)?
- Are the aggregations (SUM, AVG, COUNT) and grouping correct?
- Are filters correct? NPA / impaired = asset_classification IN ('Substandard', 'Doubtful', 'Loss').
- Does the result contain what the user asked for?

Important facts about this database (do NOT fail a query for these):
- loan_master and sector_master have NO date columns. They are a current snapshot,
  so queries on them need NO date or quarter filter.
- Only provisioning (reporting_date) and borrower_rating (rating_date) have dates.
  A "latest quarter" filter (= '2025-09-30') is needed ONLY when the question asks about
  the latest/current period AND the query uses one of those two tables.
  Questions about trends or "over the last four quarters" must NOT filter to one date;
  they should group by the date instead.
- Window functions such as SUM(...) OVER () or COUNT(*) OVER () are a valid way to show an
  overall figure (e.g., an overall percentage) next to a row-level list. No GROUP BY is needed for that.
- Only answer "no" if the query uses wrong tables/columns, wrong logic, or clearly misses
  part of what was asked.

Context about this query:
{track_context}

User question:
{user_question}

SQL query:
{candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({expected_sql}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({candidate_sql}) LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context):
    retry_prompt = f"""
You are an expert SQLite analyst. The SQL query below failed validation.
Fix it so it passes and still answers the user's original question.

Rules:
- Read the validation error carefully and fix exactly that problem.
- Return a single read-only statement that starts with SELECT or WITH. No semicolon at the end.
- Use only tables and columns from the schema.
- NPA / impaired = asset_classification IN ('Substandard', 'Doubtful', 'Loss').
- Show money amounts in USD millions with ROUND(... / 1e6, 2).
- Return ONLY the corrected SQL. No explanation, no markdown, no code fences.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    result = {'dataframe': None, 'reasonable': True, 'warnings': []}

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


def generate_response(user_question, dataframe, route, query_id=None):
    response_prompt = f"""
You are a credit risk analyst writing a short answer for bank management.
Answer the user's question using ONLY the query result below.

Rules:
- Answer exactly what was asked. If the result has more rows than needed
  (e.g., all sectors when the user asked about one), use only the relevant rows.
- Use the exact figures from the result. Never invent or estimate numbers.
- Columns ending in _mn are in USD millions; write them like "$1,195.3M".
- If useful, add one short business insight (e.g., concentration, trend, risk signal).
- Keep it to 3-6 sentences or a few short bullet points.

Route used: {route} (template: {query_id})

User question:
{user_question}

Query result:
{dataframe.to_string()}
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    log = {
        'user_question': user_question, 'route': None, 'query_id': None, 'match_reason': None,
        'candidate_sql': None, 'gate_result': None, 'retry_used': False, 'escalated': False,
        'executed_sql': None, 'row_count': None, 'confidence': None, 'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')
    if verbose:
        print(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
        print(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql
    if verbose:
        print(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate
    if verbose:
        print(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            print(f"    Failed check: {gate.get('failed_check')}")
            print(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        if verbose:
            print(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate
        if verbose:
            print(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
            if not gate['passed']:
                print(f"    Retry failed check: {gate.get('failed_check')}")
                print(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        if verbose:
            print(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    if verbose:
        print(f"[4] Execute: {len(df)} rows returned")
        if exec_result['warnings']:
            print(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative
    log['confidence'] = gate.get('relevance_confidence')
    if verbose:
        print(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}

# ---------------------------------------------------------------
# Streamlit user interface
# ---------------------------------------------------------------
EXAMPLES = [
    "How much of our book is in real estate, and how much of that is non-performing?",
    "Show me how our overdue book is distributed across DPD buckets.",
    "List every restructured loan, flag which are impaired, and show the overall impaired percentage.",
    "Show me the average interest rate by sector, sorted from highest to lowest.",
    "How has the Stage 3 book size and its expected credit loss moved over the last four quarters?",
]

st.title("📊 Credit Risk Query Engine")
st.caption("Ask a question about the loan portfolio in plain English. The engine uses a verified "
           "query when one fits, or generates and validates new SQL.")

with st.sidebar:
    st.header("Example questions")
    for i, ex in enumerate(EXAMPLES, start=1):
        if st.button(f"TC-0{i}", help=ex, use_container_width=True):
            st.session_state["question"] = ex
    st.divider()
    st.header("Verified query library")
    for qid, entry in verified_query_library.items():
        st.markdown(f"**{qid}:** {entry['description']}")
    st.divider()
    st.caption(f"Database: {db_name} (read-only) · Model: {MODEL_NAME}")

question = st.text_area("Your question", key="question", height=90,
                        placeholder="e.g., What is the total outstanding by loan category?")

if st.button("Run query", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Classifying, building SQL, validating and running..."):
            try:
                result = run_pipeline(question, conn, verified_query_library, database_schema, verbose=False)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        log = result["log"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route", str(log["route"]).capitalize())
        c2.metric("Query ID", log["query_id"] or "—")
        conf = log["confidence"]
        c3.metric("Confidence", f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf))
        c4.metric("Rows returned", log["row_count"] if log["row_count"] is not None else "—")

        if log["escalated"]:
            st.error(log["narrative"])
        else:
            st.subheader("Answer")
            st.markdown(log["narrative"])

            st.subheader("Result data")
            st.dataframe(result["dataframe"], use_container_width=True)

        with st.expander("Pipeline trace"):
            st.markdown(f"**Match reason:** {log['match_reason']}")
            st.markdown(f"**Retry used:** {log['retry_used']}")
            gate = log["gate_result"] or {}
            st.markdown(f"**Validation passed:** {gate.get('passed')}")
            if gate.get("failed_check"):
                st.markdown(f"**Failed check:** {gate.get('failed_check')}")
            st.markdown(f"**Details:** {gate.get('details')}")
            st.markdown("**SQL:**")
            st.code(log["executed_sql"] or log["candidate_sql"] or "", language="sql")
