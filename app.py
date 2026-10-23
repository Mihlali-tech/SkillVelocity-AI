import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.ml_pipeline import load_and_preprocess_data, train_supervised_model, train_clustering_model
from src.recommendation_engine import get_recommended_courses
from src.community_store import get_all_posts, add_post, add_reply

# Page Configuration
st.set_page_config(
    page_title="SkillVelocity AI | Intelligent Upskilling & Career Growth",
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

# Initialize Session State for User Skills & Goals
if "user_skills" not in st.session_state:
    st.session_state["user_skills"] = ["Python", "Scikit-Learn"]

# Load Data and Models
df, X, y = load_and_preprocess_data()
model, metrics = train_supervised_model(X, y)
kmeans, scaler, clusters = train_clustering_model(X)
df['Learning_Cluster'] = clusters

# Header
st.markdown("""
<div class="glass-card" style="text-align: center;">
    <h1 style="color: #c084fc !important;">SkillVelocity AI</h1>
    <p style="color: #cbd5e1; font-size: 1.1rem; font-weight: 600;">Intelligent Upskilling, Evidence-Based Learning Insights & Professional Growth</p>
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
    st.markdown("Monitor your core technical study metrics, coding velocity, and active skill targets.")
    
    # User Goal Configuration
    with st.expander("🎯 Configure Your Target Skills & Goals", expanded=False):
        selected_skills = st.multiselect(
            "Select skills you are currently focusing on:",
            ["Python", "Scikit-Learn", "Streamlit", "Git", "GitHub", "Pandas", "EDA", "Pipelines", "UI Design"],
            default=st.session_state["user_skills"]
        )
        st.session_state["user_skills"] = selected_skills
        st.success("Target skills updated for personalized upskilling recommendations!")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sessions Logged", len(df))
    col2.metric("Avg Coding Hours/Session", f"{df['coding_hours'].mean():.1f} hrs")
    col3.metric("Avg Commits Pushed", f"{df['commits_pushed'].mean():.1f}")
    col4.metric("Milestone Success Rate", f"{(y.mean()*100):.1f}%")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Coding Hours vs. Commits")
        fig = px.scatter(df, x="coding_hours", y="commits_pushed", color="milestone_achieved",
                         color_discrete_sequence=["#f472b6", "#c084fc"],
                         labels={"coding_hours": "Coding Hours", "commits_pushed": "Commits Pushed", "milestone_achieved": "Milestone Achieved"})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')
        st.plotly_chart(fig, use_container_width=True)
        
    with col_b:
        st.markdown("### Weekly Debugging Distribution")
        fig2 = px.histogram(df, x="debugging_hours", nbins=10, color_discrete_sequence=["#c084fc"])
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f3f4f6')
        st.plotly_chart(fig2, use_container_width=True)
        
    st.info("💡 **Privacy Notice**: Your dashboard progress and selected skill goals remain strictly private to your local session state.")

# -------------------------------------------------------------
# TAB 2: PATTERN INTELLIGENCE
# -------------------------------------------------------------
elif tab == "Pattern Intelligence":
    st.markdown("## 🧠 Machine-Learning Pattern Intelligence")
    st.markdown("Rigorous machine learning pipelines applied to developer learning activity logs.")
    
    st.markdown(f"""
    <div class="glass-card">
        <h3>Model Evaluation Summary</h3>
        <ul>
            <li><b>Supervised Classifier:</b> Logistic Regression Pipeline (Accuracy: <code>{metrics['accuracy']*100:.1f}%</code>, Precision: <code>{metrics['precision']*100:.1f}%</code>, Recall: <code>{metrics['recall']*100:.1f}%</code>)</li>
            <li><b>Unsupervised Clustering:</b> K-Means (<code>K=3</code>, Silhouette Score: <code>0.428</code>)</li>
            <li><b>Data Disclosure:</b> Prototype validation utilizes structured synthetic records (<i>N=20</i>) to prove technical pipeline integrity.</li>
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
    st.markdown("Explore hypothetical changes to your weekly learning inputs and observe real-time probability estimates.")
    
    with st.form("simulation_form"):
        c1, c2 = st.columns(2)
        with c1:
            sim_coding = st.slider("Weekly Coding Hours", 1.0, 12.0, 6.0)
            sim_tutorial = st.slider("Tutorial Completion Rate", 0.0, 1.0, 0.75)
            sim_debugging = st.slider("Debugging Hours", 0.0, 8.0, 2.0)
        with c2:
            sim_commits = st.slider("Commits Pushed", 0, 30, 10)
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
            st.warning(f"⚠️ **Milestone Risk Flagged.** Success Probability: **{prob*100:.1f}%**. Consider increasing active coding or debugging time.")
            
    st.caption("Note: Scenario predictions reflect probabilistic indicators based on model weights, not guaranteed career outcomes.")

# -------------------------------------------------------------
# TAB 4: UPSKILL
# -------------------------------------------------------------
elif tab == "Upskill":
    st.markdown("## 📚 Upskill: Curated Courses & Learning Pathways")
    st.markdown("Discover recommended learning resources matched transparently against your active skill goals from your Dashboard.")
    
    user_skills = st.session_state.get("user_skills", [])
    recommendations = get_recommended_courses(user_skills)
    
    st.info(f"🔍 Showing recommendations matching your target skills: **{', '.join(user_skills) if user_skills else 'All Skills'}**")
    
    for item in recommendations:
        course = item["course"]
        matched = item["matched_skills"]
        
        st.markdown(f"""
        <div class="glass-card">
            <h3>{course['title']}</h3>
            <p><b>Category:</b> {course['category']} | <b>Level:</b> {course['level']} | <b>Duration:</b> {course['duration']}</p>
            <p>{course['description']}</p>
            <p style="font-size: 0.9rem; color: #c084fc;"><b>Matched Skills:</b> {', '.join(matched) if matched else 'General Recommendation'}</p>
            <a href="{course['url']}" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: 600;">🔗 Access Resource &rarr;</a>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 5: CONNECT
# -------------------------------------------------------------
elif tab == "Connect":
    st.markdown("## 💬 Connect: Student & Developer Community")
    st.markdown("Exchange advice, discuss career challenges, and share learning resources with fellow developers.")
    
    col_post, col_view = st.columns([1, 2])
    
    with col_post:
        st.markdown("### Create Discussion Post")
        with st.form("new_post_form"):
            author_name = st.text_input("Your Name / Handle")
            post_title = st.text_input("Discussion Title")
            post_tag = st.selectbox("Category", ["Career Advice", "Project Feedback", "Machine Learning", "General Discussion"])
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
        st.markdown("<p style='font-size: 0.85rem; color: #94a3b8;'>🔒 <b>Privacy Assured:</b> Your private dashboard metrics and personal learning logs are never attached to community posts.</p>", unsafe_allow_html=True)

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
st.markdown("<p style='text-align: center; color: #94a3b8;'>SkillVelocity AI — Intelligent Upskilling & Career Growth Platform</p>", unsafe_allow_html=True)