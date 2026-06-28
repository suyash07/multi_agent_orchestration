Question 1: Executive overview with urgency
We're heading into Q4 board review. Give me the current state 
of customer churn, how many customers are we at risk of losing, 
and what is the single biggest driver I should be worried about?
Tests: portfolio summary + churn drivers + synthesis into executive language.

Question 2: Operational prioritization
I have a retention team of 5 reps who can each call 3 customers 
today. Give me the 15 customers they should call first and tell 
me what each rep should focus on in the conversation.
Tests: high risk customers + churn drivers + personalized talking points per customer. Forces the agent to combine two tools and reason about operational constraints.

Question 3: Geographic business problem
Our Germany regional manager is asking why his customers are 
leaving at twice the rate of other regions. What would you 
tell him and what should he do about it specifically?
Tests: churn drivers with germany segment + feature importance + recommendation synthesis. Forces it to go beyond the numbers into root cause reasoning.

Question 4: Single customer decision
I have a customer on the line right now. She is a 52 year old 
female from Germany, has been with us 3 years, balance of 
$180,000, holds 3 products, and has been inactive for several 
months. Should I offer her a retention bonus and how urgent is this?
Tests: predict_churn with specific profile + risk threshold reasoning + actionable recommendation under time pressure. The "on the line right now" framing tests urgency handling.

Question 5: Strategic product question
Our product team wants to cross-sell more products to existing 
customers to increase revenue. Based on the churn data, is this 
a good idea and what would you recommend they do instead?

Question 6: Revenue at risk
Before I walk into the CFO meeting, I need one number. 
If our high risk customers leave, what is the scale of 
the problem? How many customers are we talking about 
and should the CFO be worried?
Tests whether the agent can translate churn probability into business magnitude. Forces it to combine portfolio summary with risk distribution and frame it in financial stakes language, not data science language.

Question 7: Retention ROI logic
My CFO is asking why we should invest in a retention program 
at all. She wants to know which customers are actually worth 
saving. Are all high risk customers equal or should we be 
selective about who we spend money retaining?
This is a trap question. The agent should reason that not all high risk customers have equal value. A high risk customer with $180k balance is worth far more to retain than one with $0 balance. Tests whether it goes beyond churn probability to think about customer value segmentation.

Question 8: Counterintuitive finding
My head of customer success just told me that our most 
engaged long-term customers are the ones most likely to 
leave. She thinks the data is wrong. Is she right or wrong 
and can you explain why this might actually make sense?
References the Age and Balance paradox you discovered: older, higher-balance customers churn more despite being seemingly loyal. Tests whether the agent can defend a counterintuitive finding with evidence and explain the competitive dynamics behind it (competitors targeting high-value customers).

Question 9: Board level strategic decision
We have budget to do exactly one thing this quarter to 
reduce churn. Based on everything you know about our 
customer base, what is the single highest impact action 
we should take and why? I need to defend this to the board 
next week.