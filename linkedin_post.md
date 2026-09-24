# LinkedIn Post Draft — Customer Churn Analysis

---

Did you know acquiring a new customer costs **5x to 25x more** than retaining an existing one? 💡

In subscription and SaaS business models, customer churn isn't just a metric — it’s a direct hit to Monthly Recurring Revenue (MRR). A modest 5% increase in customer retention can boost profits by **25% to 95%**!

I recently completed an end-to-end **Customer Churn Analysis & Machine Learning Project** designed to predict high-risk accounts *before* they cancel, allowing Customer Success teams to take proactive action.

Here’s a breakdown of what I built and discovered:

🔍 **Key Behavioral Data Insights:**
- 💳 **Payment Delays**: Delays exceeding 15 days correlated with over a 60% probability of churn.
- 📞 **Support Call Volume**: Customers filing >4 support tickets exhibited a 3x higher churn rate (indicating product friction).
- 📜 **Contract Types**: Monthly subscribers churn at ~38%, whereas Annual contract holders churn at only ~10%.

🤖 **Machine Learning & Engineering Highlights:**
- **Class Imbalance Mitigation**: Avoided the "Accuracy Trap" (where predicting 'No Churn' for everyone gives false 80% accuracy) by optimizing for **Recall** and **ROC-AUC**.
- **Model Tuning**: Evaluated Decision Trees, Logistic Regression, and Random Forest using cost-sensitive loss (`class_weight='balanced'`) and probability threshold tuning.
- **Top Champion Model**: **Random Forest** achieved an **0.807 ROC-AUC** score with a **74.7% Recall**, capturing the vast majority of at-risk customers.

📊 **Business Action Plan:**
Created a tiered intervention workflow where predicted risk scores determine the retention strategy:
- 🔴 High Risk (>70%): Dedicated CSM outreach + renewal discount.
- 🟡 Medium Risk (40-70%): Automated product training workflows & payment reminders.
- 🟢 Low Risk (<40%): Standard engagement.

Check out the full repository, executed Jupyter Notebook, visual charts, and technical interview guide here:
🔗 **GitHub Repository**: https://github.com/armnJs/Customer-churn-analysis.git

I’d love to hear your thoughts! How does your team tackle customer churn? 👇

---

#DataScience #MachineLearning #Python #CustomerChurn #PredictiveAnalytics #ScikitLearn #DataAnalytics #SaaS #CustomerRetention #AI
