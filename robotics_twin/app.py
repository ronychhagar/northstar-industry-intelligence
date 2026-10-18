"""Northstar Industry Intelligence: a research-as-a-service workspace."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from robotics_twin.demo import make_demo_drive
from robotics_twin.odometry import fit_drive_model, load_sensor_csv, simulate_drive

DB_PATH = Path(__file__).resolve().parent.parent / "automotive_research.db"
STATUSES = ("Received", "Scoping", "In research", "Ready for review", "Delivered")
INDUSTRIES = ("Automotive", "Robotics", "Semiconductors", "Embedded systems")
INDUSTRY_RESEARCH = {
    "Automotive": {
        "focus": ["Market opportunity", "Competitor intelligence", "Technology & supply chain", "Customer & channel", "Product strategy", "Other"],
        "lenses": ["Electric vehicles & charging", "Software-defined vehicle", "ADAS & autonomy", "Commercial fleets", "Battery supply chain", "Aftermarket"],
        "signals": [["Demand", "Purchase intent / order pipeline", "Volume · segment · geography", "Monthly"], ["Economics", "Total cost of ownership", "Vehicle class · usage profile", "Quarterly"], ["Policy", "Incentives & compliance", "Jurisdiction · effective date", "As updated"], ["Infrastructure", "Charging / service coverage", "Location · utilization", "Quarterly"]],
        "competitor": "e.g. electric light commercial vehicles in India",
        "tech_prompt": "e.g. LFP battery systems, zonal architecture, ADAS compute",
        "comparison": "Portfolio & positioning · Price and total cost of ownership · Performance and feature set · Distribution and service · Customer segments · Partnerships and supply chain · Launches and strategic moves",
        "factors": ["Technical performance", "Unit economics / cost curve", "Supply concentration", "Regulatory / safety path", "Integration & talent", "Customer value"],
    },
    "Robotics": {
        "focus": ["Market & applications", "Robot / vendor landscape", "Components & supply chain", "Deployment economics", "Technology & autonomy", "Other"],
        "lenses": ["Industrial automation", "Collaborative robots", "Mobile robots & AMRs", "Humanoids", "Warehouse automation", "Surgical & service robots"],
        "signals": [["Adoption", "Installed base / deployments", "Application · robot type · region", "Quarterly"], ["Economics", "Payback and utilization", "Task · shift pattern · labor cost", "Per deployment"], ["Capability", "Task success / uptime", "Environment · payload · cycle time", "Per release"], ["Integration", "Deployment time and effort", "Site · integrator · workflow", "Per project"]],
        "competitor": "e.g. autonomous mobile robots for warehouse picking",
        "tech_prompt": "e.g. manipulation foundation models, lidar, safety-rated control",
        "comparison": "Robot form factor & payload · Task performance · Autonomy & safety · Integration effort · Fleet orchestration · Service model · Deployment economics · Ecosystem partners",
        "factors": ["Task capability & reliability", "Safety & certification", "Deployment / integration effort", "Unit economics & payback", "Compute, sensors & components", "Serviceability & fleet tools"],
    },
    "Semiconductors": {
        "focus": ["Market & end-market demand", "Chip / supplier landscape", "Process & packaging technology", "Capacity & supply chain", "Design-win intelligence", "Other"],
        "lenses": ["AI accelerators", "Automotive semiconductors", "Power & analog", "Memory", "Advanced packaging", "Fab equipment & materials"],
        "signals": [["Demand", "Unit shipments / design wins", "Node · application · geography", "Quarterly"], ["Capacity", "Fab and packaging utilization", "Node · site · process step", "Quarterly"], ["Economics", "ASP / wafer cost trends", "Product · node · volume band", "Quarterly"], ["Supply", "Lead times and concentration", "Component · supplier · region", "Monthly"]],
        "competitor": "e.g. edge AI inference accelerators for industrial devices",
        "tech_prompt": "e.g. chiplets, HBM, GAA nodes, silicon carbide power devices",
        "comparison": "Process node & product portfolio · Performance / power / area · Packaging & memory · Foundry / OSAT capacity · Design ecosystem · End-market exposure · Roadmap · Export and supply constraints",
        "factors": ["Performance, power & area", "Process maturity & yield", "Packaging / memory dependencies", "Capacity & geographic concentration", "Design ecosystem & IP", "Cost, lead time & qualification"],
    },
    "Embedded systems": {
        "focus": ["Platform & architecture", "RTOS / software ecosystem", "Compute & connectivity", "Safety & security", "Supplier landscape", "Other"],
        "lenses": ["Edge AI", "RTOS & embedded Linux", "MCUs & SoCs", "Industrial IoT", "Connectivity", "Functional safety & cybersecurity"],
        "signals": [["Adoption", "Platform wins / design starts", "Application · processor · region", "Quarterly"], ["Lifecycle", "Availability and longevity", "Part · supplier · lifecycle status", "Monthly"], ["Engineering", "BOM, power and development effort", "Workload · volume · certification", "Per program"], ["Risk", "Vulnerabilities and compliance", "CVE · standard · product family", "As updated"]],
        "competitor": "e.g. safety-certified real-time operating systems for industrial control",
        "tech_prompt": "e.g. RISC-V MCUs, TSN Ethernet, edge AI runtimes, secure boot",
        "comparison": "Compute & memory · OS / RTOS support · Toolchain & middleware · Connectivity · Safety / security certifications · Lifecycle policy · Power & BOM · Vendor support",
        "factors": ["Compute, memory & power envelope", "OS / RTOS and middleware support", "Toolchain & developer ecosystem", "Safety / security evidence", "Lifecycle, availability & longevity", "BOM cost & integration effort"],
    },
}


def _db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE IF NOT EXISTS research_requests (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, company TEXT NOT NULL,
            contact TEXT NOT NULL, title TEXT NOT NULL, decision TEXT NOT NULL,
            segment TEXT NOT NULL, markets TEXT NOT NULL, competitors TEXT NOT NULL,
            deliverable TEXT NOT NULL, due_date TEXT NOT NULL, status TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '')"""
    )
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(research_requests)")}
    if "industry" not in columns:
        connection.execute("ALTER TABLE research_requests ADD COLUMN industry TEXT NOT NULL DEFAULT 'Automotive'")
    connection.commit()
    return connection


def _requests() -> pd.DataFrame:
    with _db() as connection:
        return pd.read_sql_query(
            "SELECT * FROM research_requests ORDER BY created_at DESC", connection
        )


def _create_request(values: dict[str, str]) -> None:
    with _db() as connection:
        connection.execute(
            """INSERT INTO research_requests
            (id, created_at, company, contact, title, decision, industry, segment, markets,
             competitors, deliverable, due_date, status, notes)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (uuid.uuid4().hex[:8].upper(), datetime.now().isoformat(timespec="minutes"),
             values["company"], values["contact"], values["title"], values["decision"],
             values["industry"], values["segment"], values["markets"], values["competitors"],
             values["deliverable"], values["due_date"], "Received", values["notes"]),
        )


def _update_request(request_id: str, status: str, notes: str) -> None:
    with _db() as connection:
        connection.execute(
            "UPDATE research_requests SET status=?, notes=? WHERE id=?",
            (status, notes, request_id),
        )


def _style() -> None:
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      .stApp { background:#f4f5f2; color:#17221f; }
      [data-testid="stSidebar"] { background:#172622; }
      [data-testid="stSidebar"] * { color:#e7ede7 !important; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#9baea4 !important; }
      h1,h2,h3 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
      h1 { font-size:2.3rem !important; }
      [data-testid="stMetric"] { background:#fff; border:1px solid #e6e9e3; padding:18px 20px; border-radius:12px; }
      [data-testid="stMetricLabel"] { color:#687570; }
      .eyebrow { font:500 11px 'DM Mono',monospace; letter-spacing:.12em; color:#658177; text-transform:uppercase; }
      .hero { background:#1c302a; color:#f6f7f1; padding:32px 34px; border-radius:16px; margin:8px 0 24px; }
      .hero h1 { color:#f6f7f1; margin:7px 0 8px; }
      .hero p { color:#bac9c0; max-width:760px; font-size:15px; }
      .panel { background:white; border:1px solid #e5e9e3; border-radius:12px; padding:20px 22px; }
      .signal { border-left:3px solid #94ad45; padding:2px 0 2px 13px; margin:15px 0; }
      .signal b { color:#1c302a; }
      .signal small { color:#708078; }
      .tag { display:inline-block; background:#eaf0e5; color:#526a37; border-radius:99px; padding:4px 9px; font:11px 'DM Mono',monospace; }
      div.stButton > button[kind="primary"] { background:#9dbb4b; border:0; color:#17221f; font-weight:700; }
      [data-testid="stForm"] { background:#fff; padding:22px; border:1px solid #e6e9e3; border-radius:12px; }
    </style>""", unsafe_allow_html=True)


def _header(section: str, sub: str) -> None:
    st.markdown(f'<div class="eyebrow">{section}</div>', unsafe_allow_html=True)
    st.title(sub)


def _overview(requests: pd.DataFrame, industry: str) -> None:
    st.markdown(f"""
    <div class="hero"><div class="eyebrow" style="color:#b1c765">{industry.upper()} INTELLIGENCE · CLIENT WORKSPACE</div>
    <h1>Research that moves decisions forward.</h1>
    <p>One research desk for market entry, competitor moves, technology shifts, and the questions your next product decision depends on.</p></div>""", unsafe_allow_html=True)
    total = len(requests)
    active = int((requests.status != "Delivered").sum()) if total else 0
    delivered = int((requests.status == "Delivered").sum()) if total else 0
    a, b, c, d = st.columns(4)
    a.metric("Research requests", total, help="Requests stored in this workspace")
    b.metric("Active engagements", active)
    c.metric("Delivered", delivered)
    d.metric("Research coverage", "4 lenses", help="Markets · competitors · technology · customer")
    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Your research desk")
        if requests.empty:
            st.markdown('<div class="panel"><span class="eyebrow">START WITH A DECISION</span><h3>Turn a strategic question into a research brief.</h3><p>Tell us what you need to decide, which market matters, and when you need an answer. Your team can track the engagement here.</p></div>', unsafe_allow_html=True)
            if st.button("Create a research request", type="primary"):
                st.session_state.page = "Research requests"
                st.rerun()
        else:
            st.dataframe(requests[["id", "industry", "title", "company", "segment", "status", "due_date"]],
                         hide_index=True, use_container_width=True,
                         column_config={"id":"ID", "industry":"Industry", "title":"Research question", "company":"Company",
                                        "segment":"Focus area", "status":"Status", "due_date":"Target date"})
        st.caption("Engagement tracking and saved requests are stored locally in this workspace.")
    with right:
        st.subheader("Research capabilities")
        items = [("01", "Market & opportunity", "Sizing, adoption signals, whitespace and go-to-market."),
                 ("02", "Competitive intelligence", "Product launches, positioning, pricing and strategic moves."),
                 ("03", "Technology & supply chain", f"{industry} platforms, components, suppliers, and ecosystem shifts."),
                 ("04", "Customer & deployment", "Buyer needs, adoption barriers, economics, and implementation dynamics.")]
        for no, title, desc in items:
            st.markdown(f'<div class="signal"><small>{no} / CAPABILITY</small><br><b>{title}</b><br><small>{desc}</small></div>', unsafe_allow_html=True)
    st.info("This workspace tracks research engagements. Market findings and estimates should be added with cited, dated evidence before being used for business decisions.")


def _requests_page(requests: pd.DataFrame, active_industry: str) -> None:
    _header("ENGAGEMENTS", "Research requests")
    st.write("Scope a decision-led brief, track its progress, and export the request for your research team.")
    with st.form("research_request", clear_on_submit=True):
        st.subheader("New research brief")
        col1, col2 = st.columns(2)
        company = col1.text_input("Company", placeholder="Company or business unit")
        contact = col2.text_input("Request owner", placeholder="Name or work email")
        industry = st.selectbox("Industry", INDUSTRIES, index=INDUSTRIES.index(active_industry))
        title = st.text_input("What do you need to know?", placeholder="e.g. Where should we launch our next-generation fleet charging offer?")
        decision = st.text_area("Decision this research will inform", placeholder="What will you do differently once you have the answer?", height=85)
        col1, col2 = st.columns(2)
        segment = col1.selectbox("Research focus", INDUSTRY_RESEARCH[industry]["focus"])
        markets = col2.text_input("Markets / geographies", placeholder="e.g. India, EU, North America")
        competitors = st.text_input("Companies or products to include", placeholder="Optional · separate with commas")
        col1, col2 = st.columns(2)
        deliverable = col1.selectbox("Preferred deliverable", ["Executive decision brief", "Market landscape", "Competitor comparison", "Technology / supplier map", "Custom workshop", "Not sure yet"])
        due_date = col2.date_input("Target date", value=date.today())
        notes = st.text_area("Context, constraints, or source material", placeholder="Include target customer, price band, internal hypotheses, or attach context in a follow-up.", height=90)
        submitted = st.form_submit_button("Submit research request", type="primary", use_container_width=True)
        if submitted:
            if not company.strip() or not title.strip() or not decision.strip():
                st.error("Add your company, research question, and the decision this should inform.")
            else:
                _create_request({"company":company.strip(), "contact":contact.strip(), "industry":industry, "title":title.strip(),
                                 "decision":decision.strip(), "segment":segment, "markets":markets.strip(),
                                 "competitors":competitors.strip(), "deliverable":deliverable,
                                 "due_date":due_date.isoformat(), "notes":notes.strip()})
                st.success("Request saved. It is now in the engagement tracker below.")
                st.rerun()
    requests = _requests()
    st.subheader("Engagement tracker")
    if requests.empty:
        st.caption("Submitted requests will appear here.")
    else:
        visible = requests[["id", "created_at", "company", "industry", "title", "segment", "markets", "status", "due_date"]].copy()
        st.dataframe(visible, hide_index=True, use_container_width=True,
                     column_config={"id":"ID", "created_at":"Created", "company":"Company", "industry":"Industry", "title":"Research question", "segment":"Focus", "markets":"Markets", "status":"Status", "due_date":"Target date"})
        ids = requests.id.tolist()
        chosen = st.selectbox("Open a request", ids, format_func=lambda value: f"{value} · {requests.loc[requests.id == value, 'title'].iloc[0]}")
        record = requests.loc[requests.id == chosen].iloc[0]
        with st.expander("Request details & progress", expanded=True):
            st.markdown(f"**Decision:** {record.decision}\n\n**Competitors:** {record.competitors or 'Not specified'}\n\n**Deliverable:** {record.deliverable}\n\n**Context:** {record.notes or 'None provided'}")
            with st.form(f"update_{chosen}"):
                status = st.selectbox("Engagement status", STATUSES, index=STATUSES.index(record.status))
                update_notes = st.text_area("Research team notes", value=record.notes, height=80)
                save = st.form_submit_button("Save progress")
                if save:
                    _update_request(chosen, status, update_notes)
                    st.rerun()
            brief = {key: (record[key].isoformat() if hasattr(record[key], "isoformat") else str(record[key])) for key in requests.columns}
            st.download_button("Download request brief (JSON)", json.dumps(brief, indent=2), file_name=f"research-brief-{chosen}.json", mime="application/json")
        st.download_button("Export engagement tracker (CSV)", requests.to_csv(index=False), file_name="research-engagements.csv", mime="text/csv")


def _market_monitor(industry: str) -> None:
    research = INDUSTRY_RESEARCH[industry]
    _header("SIGNALS", "Market monitor")
    st.write(f"A framework for monitoring the market signals behind {industry.lower()} investment decisions.")
    st.warning("No market figures are populated in this demo. Add sourced data with a date, geography, definition, and confidence before using estimates in a decision.")
    lens = st.selectbox("Choose a research lens", research["lenses"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Market size", "—", help="Requires a sourced definition and estimate")
    c2.metric("Growth outlook", "—")
    c3.metric("Evidence coverage", "0 sources")
    st.subheader(f"{lens}: evidence workspace")
    uploaded = st.file_uploader("Import a source log (CSV)", type=["csv"], key="market_sources")
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            st.dataframe(data, use_container_width=True, hide_index=True)
            st.caption(f"{len(data):,} rows · {len(data.columns):,} fields. Check source, date, units, and geography before interpretation.")
        except Exception as error:
            st.error(f"Could not read CSV: {error}")
    else:
        st.markdown('<div class="panel"><span class="eyebrow">RESEARCH OUTPUT TEMPLATE</span><h3>Build a defensible market view</h3><p>Track the market definition, total addressable demand, growth drivers, adoption barriers, regulatory context, customer economics, and evidence gaps.</p><div class="signal"><b>Every finding should carry its source, publication date, geography, methodology, and confidence.</b></div></div>', unsafe_allow_html=True)
    st.write("")
    st.subheader("Signal watchlist")
    signals = pd.DataFrame(research["signals"], columns=["Lens", "Signal", "Required cut", "Cadence"])
    st.dataframe(signals, hide_index=True, use_container_width=True)


def _competitor_landscape(industry: str) -> None:
    _header("COMPETITIVE INTELLIGENCE", "Competitor landscape")
    st.write("Structure a competitor review around the moves that matter to your product and market position.")
    st.warning("Competitor entries are intentionally left blank. Populate with dated, attributable public or client-provided evidence.")
    markets = st.text_input("Market / product category", placeholder=INDUSTRY_RESEARCH[industry]["competitor"])
    company_list = st.text_area("Companies to compare", placeholder="Enter one company per line", height=100)
    if st.button("Build comparison matrix", type="primary"):
        companies = [item.strip() for item in company_list.splitlines() if item.strip()]
        if not companies:
            st.error("Add at least one company to create a comparison matrix.")
        else:
            matrix = pd.DataFrame({"Company":companies, "Positioning":"", "Key products":"", "Price / TCO":"", "Channels":"", "Recent moves":"", "Evidence / date":""})
            st.session_state[f"competitor_matrix_{industry}"] = matrix
    matrix_key = f"competitor_matrix_{industry}"
    if matrix_key in st.session_state:
        st.data_editor(st.session_state[matrix_key], use_container_width=True, hide_index=True, num_rows="dynamic", key=f"competitor_editor_{industry}")
        st.caption(f"Scope: {markets or 'Not specified'} · Treat unverified entries as hypotheses until sourced.")
    st.subheader("Comparison dimensions")
    st.markdown("**" + INDUSTRY_RESEARCH[industry]["comparison"].replace(" · ", "** · **") + "**")


def _technology_radar(industry: str) -> None:
    _header("TECHNOLOGY & VALUE CHAIN", "Technology radar")
    st.write("Evaluate technologies by readiness, strategic relevance, adoption friction, and ecosystem dependencies.")
    st.warning("Radar entries are a blank research template, not current technology ratings.")
    technology = st.text_input("Technology or supplier domain", placeholder=INDUSTRY_RESEARCH[industry]["tech_prompt"])
    cols = st.columns(4)
    stage = cols[0].selectbox("Readiness", ["Explore", "Validate", "Pilot", "Scale"])
    impact = cols[1].selectbox("Potential impact", ["Low", "Medium", "High", "Transformative"])
    horizon = cols[2].selectbox("Decision horizon", ["0–2 years", "2–5 years", "5+ years"])
    confidence = cols[3].selectbox("Evidence confidence", ["Low", "Medium", "High"])
    factors = pd.DataFrame([[factor, "", ""] for factor in INDUSTRY_RESEARCH[industry]["factors"]], columns=["Assessment factor", "Finding", "Source / date"])
    edited = st.data_editor(factors, use_container_width=True, hide_index=True, key=f"radar_editor_{industry}")
    if st.button("Save technology assessment"):
        st.session_state[f"technology_assessment_{industry}"] = {"industry": industry, "technology":technology, "readiness":stage, "impact":impact, "horizon":horizon, "confidence":confidence, "factors":edited.to_dict(orient="records")}
        st.success("Assessment saved for this session.")
    assessment_key = f"technology_assessment_{industry}"
    if assessment_key in st.session_state:
        st.download_button("Export assessment JSON", json.dumps(st.session_state[assessment_key], indent=2), file_name=f"technology-assessment-{industry.lower().replace(' ', '-')}.json", mime="application/json")


def _motion_lab() -> None:
    _header("ENGINEERING MODULE", "MotionLab · sensor calibration")
    st.write("Calibrate differential-drive robot motion from wheel encoders and IMU readings.")
    demo = make_demo_drive()
    wheel_radius = st.sidebar.number_input("Measured wheel radius (m)", min_value=0.005, max_value=1.0, value=0.05, step=0.005, format="%.3f")
    train_fraction = st.sidebar.slider("Training data (%)", 50, 90, 70, 5) / 100
    upload = st.file_uploader("Robot sensor log (CSV)", type=("csv",), help="Required: time_s, left_encoder_rad, right_encoder_rad, gyro_z_rad_s")
    if upload is None:
        drive_log = demo.log
        st.caption("Using a synthetic example drive. It is not real telemetry.")
        st.download_button("Download example sensor log", demo.csv_bytes, "example_robot_drive.csv", "text/csv")
    else:
        try:
            drive_log = load_sensor_csv(upload.getvalue())
        except ValueError as error:
            st.error(str(error)); return
    try:
        model = fit_drive_model(drive_log, wheel_radius, train_fraction)
        simulation = simulate_drive(drive_log, model)
    except ValueError as error:
        st.error(str(error)); return
    a, b, c = st.columns(3)
    a.metric("Estimated track width", f"{model.track_width_m:.3f} m")
    b.metric("Training yaw-rate RMSE", f"{model.training_gyro_rmse_rad_s:.4f} rad/s")
    c.metric("Held-out yaw-rate RMSE", f"{model.validation_gyro_rmse_rad_s:.4f} rad/s")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(simulation.x_m, simulation.y_m, color="#526d47", lw=2)
    ax.set(title="Encoder-reconstructed path", xlabel="East (m)", ylabel="North (m)", aspect="equal")
    ax.grid(alpha=.25); fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    st.caption("Assumes synchronized sensors, planar motion, and no wheel slip. Reconstructed path is odometry, not ground truth.")


def main() -> None:
    st.set_page_config(page_title="Northstar | Industry Intelligence", page_icon="◈", layout="wide")
    _style()
    requests = _requests()
    with st.sidebar:
        st.markdown("<div style='font:800 19px Manrope;color:#eff3ea'>NORTHSTAR<span style='color:#a9c358'> /</span></div><div style='font:11px DM Mono;color:#9baea4;letter-spacing:.12em;margin:3px 0 22px'>INDUSTRY INTELLIGENCE</div>", unsafe_allow_html=True)
        st.markdown("<div class='eyebrow'>WORKSPACE</div>", unsafe_allow_html=True)
        industry = st.selectbox("Industry workspace", INDUSTRIES, key="industry_workspace")
        pages = ["Overview", "Research requests", "Market monitor", "Competitor landscape", "Technology radar", "MotionLab"]
        current = st.radio("Workspace navigation", pages, label_visibility="collapsed", index=pages.index(st.session_state.get("page", "Overview")) if st.session_state.get("page", "Overview") in pages else 0)
        st.session_state.page = current
        st.divider()
        st.markdown("<div class='eyebrow'>RESEARCH DESK</div>", unsafe_allow_html=True)
        st.caption("Decision-led research · Evidence-first · Domain specific")
        st.caption(f"{len(requests)} saved engagement{'s' if len(requests) != 1 else ''}")
    if current == "Overview": _overview(requests, industry)
    elif current == "Research requests": _requests_page(requests, industry)
    elif current == "Market monitor": _market_monitor(industry)
    elif current == "Competitor landscape": _competitor_landscape(industry)
    elif current == "Technology radar": _technology_radar(industry)
    else: _motion_lab()


if __name__ == "__main__":
    main()
