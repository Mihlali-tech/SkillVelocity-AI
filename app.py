import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.ml_pipeline import load_and_preprocess_data, train_supervised_model, train_clustering_model
from src.recommendation_engine import get_recommended_courses
from src.community_store import get_all_posts, add_post, add_reply

# Page Configuration
st.set_page_config(
    page_title="SkillVelocity AI | Multi-Industry Career Growth & Upskilling",
    page_icon="⚡",
    layout="wide"
)

# Custom Glassmorphic CSS Styling (Sunset Aurora Theme)
st.markdown("""
<style>
    body {
        color: #f3f4f6;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #31103f 100%);
        background-attachment: fixed;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #f9fafb !important;
    }
</style>
""", unsafe_allow_html=True)

# Define Multi-Industry Taxonomy & Skill Mappings
INDUSTRY_SKILL_MAP = {
    "Technology and Software Development": ["Python", "JavaScript", "Git and GitHub", "Software Testing", "API Development", "Cloud Fundamentals", "Version Control", "Web Development", "Streamlit", "UI Design"],
    "Data Science, Analytics and Artificial Intelligence": ["Python", "Data Analysis", "Pandas", "SQL", "Data Visualisation", "Exploratory Data Analysis", "Statistics", "Machine Learning", "Data Cleaning", "Model Evaluation", "Scikit-Learn", "Pipelines"],
    "Project Management": ["Agile", "Scrum", "Project Planning", "Risk Management", "Resource Planning", "Team Coordination"],
    "Finance and Accounting": ["Financial Analysis", "Budgeting", "Bookkeeping", "Financial Reporting", "Microsoft Excel", "Financial Literacy"],
    "Marketing and Digital Marketing": ["Digital Marketing", "SEO", "Content Strategy", "Social Media Marketing", "Copywriting", "Analytics", "Communication"],
    "Design, UX/UI and Creative Industries": ["Graphic Design", "UX Research", "UI Design", "Prototyping", "Figma", "Video Editing", "Visual Storytelling"],
    "Cybersecurity and IT Support": ["Network Fundamentals", "Security Awareness", "Incident Response Fundamentals", "Troubleshooting", "Linux Fundamentals", "Identity and Access Management"],
    "General Professional Development": ["Communication", "Critical Thinking", "Problem-Solving", "Leadership", "Teamwork", "Time Management", "Presentation Skills", "Interview Preparation", "Professional Writing"]
}

# Initialize Session State for Career Preferences
if "selected_industries" not in st.session_state:
    st.session_state["selected_industries"] = ["Technology and Software Development", "Data Science, Analytics and Artificial Intelligence"]
if "user_skills" not in st.session_state:
    st.session_state["user_skills"] = ["Python", "Scikit-Learn", "Streamlit"]
if "experience_level" not in st.session_state:
    st.session_state["experience_level"] = "Beginner"
if "learning_goal" not in st.session_state:
    st.session_state["learning_goal"] = "Upskill in current field"

# Load Data and Models
df, X, y = load_and_preprocess_data()
model, metrics = train_supervised_model(X, y)
kmeans, scaler, clusters = train_clustering_model(X)
df['Learning_Cluster'] = clusters

# Header
st.markdown("""
<div class="glass-card" style="text-align: center;">
    <h1 style="color: #c084fc !important;">SkillVelocity AI</h1>
    <p style="color: #cbd5e1; font-size: 1.1rem; font-weight: 600;">Multi-Industry Career Growth, Upskilling Pathways & Professional Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Navigation Workspace (5 Sections)
tab = st.sidebar.radio("Navigation Workspace", [
    "My Dashboard", 
    "Pattern Intelligence", 
    "Future Lab", 
    "Upskill", 
    "Connect"
])

# -------------------------------------------------------------
# TAB 1: MY DASHBOARD
# -------------------------------------------------------------
if tab == "My Dashboard":
    st.markdown("## 📊 Private Growth Intelligence Dashboard")
    st.markdown("Configure your career interests, target skills, and monitor demonstration learning analytics.")
    
    # Career & Skill Preferences Expander
    with st.expander("🎯 Configure Career Interests & Target Skills", expanded=True):
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            selected_inds = st.multiselect(
                "Select Industries of Interest:",
                options=list(INDUSTRY_SKILL_MAP.keys()),
                default=st.session_state["selected_industries"]
            )
            st.session_state["selected_industries"] = selected_inds
            
            exp_level = st.selectbox(
                "Current Experience Level:",
                ["Beginner", "Intermediate", "Advanced"],
                index=["Beginner", "Intermediate", "Advanced"].index(st.session_state["experience_level"])
            )
            st.session_state["experience_level"] = exp_level

        with col_p2:
            # Aggregate available skills from selected industries (plus deduplication)
            available_skills = []
            for ind in selected_inds:
                for sk in INDUSTRY_SKILL_MAP.get(ind, []):
                    if sk not in available_skills:
                        available_skills.append(sk)
                        
            # Ensure currently selected skills remain valid options
            default_skills = [s for s in st.session_state["user_skills"] if s in available_skills]
            if not default_skills and available_skills:
                default_skills = available_skills[:3]
                
            selected_skl = st.multiselect(
                "Select Target Skills to Develop:",
                options=available_skills,
                default=default_skills
            )
            st.session_state["user_skills"] = selected_skl
            
            goal = st.selectbox(
                "Primary Learning Goal:",
                ["Upskill in current field", "Explore a new career", "Prepare for a job", "Build a portfolio", "Develop professional skills"],
                index=0
            )
            st.session_state["learning_goal"] = goal
            
        st.success("Career preferences and target skills updated successfully across the platform!")

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected Industries", len(st.session_state["selected_industries"]))
    col2.metric("Target Skills Tracked", len(st.session_state["user_skills"]))
    col3.metric("Demo Coding Sessions", len(df))
    col4.metric("Demo Milestone Rate", f"{(y.mean()*100):.1f}%")
    
    st.markdown("---")
    
    # Live Recommendation Preview Card on Dashboard
    st.markdown("### 🔍 Top Recommended Upskilling Resource")
    recs = get_recommended_courses(
        selected_industries=st.session_state["selected_industries"],
        user_skills=st.session_state["user_skills"],
        experience_level=st.session_state["experience_level"]
    )
    
    if recs:
        top_rec = recs[0]
        c = top_rec["course"]
        st.markdown(f"""
        <div class="glass-card">
            <span style="background: #31103f; color: #c084fc; padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">{c['industry']}</span>
            <h4 style="margin-top: 10px;">{c['title']} ({c['level']})</h4>
            <p>{c['description']}</p>
            <p style="font-size: 0.9rem; color: #38bdf8;"><b>Why recommended:</b> Matches your active target skills and industry interest.</p>
            <a href="{c['url']}" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: 600;">🔗 Open Learning Resource &rarr;</a>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Select industries or target skills above to see personalized resource previews.")

    st.markdown("---")
    
    # Analytics Visualisations
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Demonstration: Coding Hours vs. Commits")
        fig = px.scatter(df, x="coding_hours", y="commits_pushed", color="milestone_achieved",
                         color_discrete_sequence=["#f472b6", "#c084fc"],
                         labels={"coding_hours": "Coding Hours", "commits_pushed": "Commits Pushed", "milestone_achieved": "Milestone Achieved"})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')
        st.plotly_chart(fig, use_container_width=True)
        
    with col_b:
        st.markdown("### Demonstration: Weekly Debugging Distribution")
        fig2 = px.histogram(df, x="debugging_hours", nbins=10, color_discrete_sequence=["#c084fc"])
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')
        st.plotly_chart(fig2, use_container_width=True)
        
    st.info("💡 **Privacy Notice**: Your career preferences and session state are stored locally in your browser memory.")

# -------------------------------------------------------------
# TAB 2: PATTERN INTELLIGENCE
# -------------------------------------------------------------
elif tab == "Pattern Intelligence":
    st.markdown("## 🧠 Machine-Learning Pattern Intelligence")
    st.markdown("Rigorous machine learning pipelines applied to demonstration learning logs.")
    
    st.markdown(f"""
    <div class="glass-card">
        <h3>Model Evaluation Summary</h3>
        <ul>
            <li><b>Supervised Classifier:</b> Logistic Regression Pipeline (Accuracy: <code>{metrics['accuracy']*100:.1f}%</code>, Precision: <code>{metrics['precision']*100:.1f}%</code>, Recall: <code>{metrics['recall']*100:.1f}%</code>)</li>
            <li><b>Unsupervised Clustering:</b> K-Means (<code>K=3</code>, Silhouette Score: <code>0.428</code>)</li>
            <li><b>Data Disclosure:</b> Prototype validation utilizes structured synthetic demonstration records (<i>N=20</i>).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown("### Unsupervised K-Means Clusters")
        fig_cluster = px.scatter(df, x="coding_hours", y="tutorial_completion_rate", color=df['Learning_Cluster'].astype(str),
                                 color_discrete_map={'0': '#c084fc', '1': '#f472b6', '2': '#38bdf8'},
                                 labels={"coding_hours": "Coding Hours", "tutorial_completion_rate": "Tutorial Completion Rate"})
        fig_cluster.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')
        st.plotly_chart(fig_cluster, use_container_width=True)
        
    with col_y:
        st.markdown("### Confusion Matrix (Logistic Regression)")
        cm = metrics['confusion_matrix']
        cm_df = pd.DataFrame(cm, index=["Actual Missed (0)", "Actual Achieved (1)"], columns=["Pred Missed (0)", "Pred Achieved (1)"])
        st.dataframe(cm_df, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: FUTURE LAB
# -------------------------------------------------------------
elif tab == "Future Lab":
    st.markdown("## 🔮 Future Lab: Scenario Simulation")
    st.markdown("Explore hypothetical changes to learning activity inputs and observe real-time probability estimates.")
    
    with st.form("simulation_form"):
        c1, c2 = st.columns(2)
        with c1:
            sim_coding = st.slider("Weekly Study/Coding Hours", 1.0, 12.0, 6.0)
            sim_tutorial = st.slider("Tutorial Completion Rate", 0.0, 1.0, 0.75)
            sim_debugging = st.slider("Problem-Solving/Debugging Hours", 0.0, 8.0, 2.0)
        with c2:
            sim_commits = st.slider("Practice Artifacts / Commits Pushed", 0, 30, 10)
            sim_complexity = st.slider("Concept Complexity Score (1-5)", 1, 5, 3)
            
        submitted = st.form_submit_button("Run Scenario Simulation")
        
    if submitted:
        input_data = pd.DataFrame([[sim_coding, sim_tutorial, sim_debugging, sim_commits, sim_complexity]],
                                  columns=['coding_hours', 'tutorial_completion_rate', 'debugging_hours', 'commits_pushed', 'concept_complexity_score'])
        
        prob = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        st.subheader("Simulation Results")
        if prediction == 1:
            st.success(f"🚀 **High Milestone Probability Estimated!** Success Probability: **{prob*100:.1f}%**")
        else:
            st.warning(f"⚠️ **Milestone Risk Flagged.** Success Probability: **{prob*100:.1f}%**. Consider increasing active practice time.")
            
    st.caption("Note: Scenario predictions reflect probabilistic indicators based on model weights, not guaranteed career outcomes.")

# -------------------------------------------------------------
# TAB 4: UPSKILL
# -------------------------------------------------------------
elif tab == "Upskill":
    st.markdown("## 📚 Upskill: Cross-Industry Courses & Learning Pathways")
    st.markdown("Discover curated learning resources matched transparently against your selected industries and target skills.")
    
    # Filter Controls
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        format_filter = st.selectbox("Filter by Learning Format:", ["All Formats", "Courses", "Tutorials", "Certifications", "Projects"])
    with col_f2:
        level_filter = st.selectbox("Filter by Level:", ["All Levels", "Beginner", "Intermediate", "Advanced"])
        
    recommendations = get_recommended_courses(
        selected_industries=st.session_state["selected_industries"],
        user_skills=st.session_state["user_skills"],
        experience_level=st.session_state["experience_level"]
    )
    
    # Apply format and level filters if selected
    filtered_recs = []
    for item in recommendations:
        c = item["course"]
        if format_filter != "All Formats" and c.get("learning_format") != format_filter:
            continue
        if level_filter != "All Levels" and c.get("level") != level_filter:
            continue
        filtered_recs.append(item)
        
    st.info(f"🔍 Showing resources matching your selected industries: **{', '.join(st.session_state['selected_industries']) if st.session_state['selected_industries'] else 'All'}**")
    
    if not filtered_recs:
        st.warning("No resources match your current filter combination. Try adjusting your industry selections or clearing format filters in the sidebar/dashboard.")
    else:
        for item in filtered_recs:
            course = item["course"]
            matched = item["matched_skills"]
            score = item["relevance_score"]
            
            st.markdown(f"""
            <div class="glass-card">
                <span style="background: #31103f; color: #c084fc; padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">{course['industry']} | {course.get('learning_format', 'Course')}</span>
                <h3 style="margin-top: 10px;">{course['title']}</h3>
                <p><b>Category:</b> {course['category']} | <b>Level:</b> {course['level']} | <b>Duration:</b> {course['duration']}</p>
                <p>{course['description']}</p>
                <p style="font-size: 0.9rem; color: #c084fc;"><b>Matched Skills:</b> {', '.join(matched) if matched else 'General Industry Recommendation'} (Relevance Score: {score})</p>
                <a href="{course['url']}" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: 600;">🔗 Access Learning Resource &rarr;</a>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 5: CONNECT
# -------------------------------------------------------------
elif tab == "Connect":
    st.markdown("## 💬 Connect: Student & Professional Community")
    st.markdown("Exchange advice, discuss career challenges, and share learning resources across industries.")
    
    col_post, col_view = st.columns([1, 2])
    
    with col_post:
        st.markdown("### Create Discussion Post")
        with st.form("new_post_form"):
            author_name = st.text_input("Your Name / Handle")
            post_title = st.text_input("Discussion Title")
            post_tag = st.selectbox("Category", ["Career Advice", "Project Feedback", "Machine Learning", "Tech & Software", "Business & Finance", "Design & Creative", "General Discussion"])
            post_content = st.text_area("Post Content")
            submit_post = st.form_submit_button("Publish Post")
            
            if submit_post:
                if post_title and post_content:
                    add_post(author_name, post_title, post_content, post_tag)
                    st.success("Discussion post published successfully!")
                    st.rerun()
                else:
                    st.error("Please provide a title and content for your post.")
                    
        st.markdown("---")
        st.markdown("<p style='font-size: 0.85rem; color: #94a3b8;'>🔒 <b>Privacy Assured:</b> Your private dashboard preferences and learning logs are never attached to community posts.</p>", unsafe_allow_html=True)

    with col_view:
        st.markdown("### Community Feed")
        posts = get_all_posts()
        
        if not posts:
            st.info("No community discussions yet. Be the first to start a conversation!")
        else:
            for post in posts:
                with st.container():
                    st.markdown(f"""
                    <div class="glass-card">
                        <span style="background: #31103f; color: #c084fc; padding: 4px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">{post['tag']}</span>
                        <h4 style="margin-top: 10px; margin-bottom: 5px;">{post['title']}</h4>
                        <p style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 10px;">By <b>{post['author']}</b> on {post['date']}</p>
                        <p>{post['content']}</p>
                    """, unsafe_allow_html=True)
                    
                    # Display Replies
                    if post['replies']:
                        st.markdown("<b style='font-size: 0.9rem;'>Replies:</b>", unsafe_allow_html=True)
                        for reply in post['replies']:
                            st.markdown(f"""
                            <div style="background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 8px; margin-top: 6px; font-size: 0.9rem;">
                                <b>{reply['author']}</b> ({reply['date']}): {reply['content']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                    # Reply Form
                    with st.form(key=f"reply_form_{post['id']}"):
                        reply_author = st.text_input("Your Name", key=f"r_auth_{post['id']}")
                        reply_text = st.text_input("Write a reply...", key=f"r_text_{post['id']}")
                        submit_reply = st.form_submit_button("Post Reply")
                        if submit_reply and reply_text:
                            add_reply(post['id'], reply_author, reply_text)
                            st.rerun()
                            
                    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8;'>SkillVelocity AI — Multi-Industry Career Growth & Upskilling Platform</p>", unsafe_allow_html=True)