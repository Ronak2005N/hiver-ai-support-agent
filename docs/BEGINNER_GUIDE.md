# Hiver Project - Step-by-Step Guide for Beginners

## What We're Building (Simple Explanation)

Imagine you work at Apple. Every day, thousands of people tweet at @Apple saying things like:
- "My iPhone is broken"
- "I was charged twice"
- "Where is my order?"

Right now, HUMANS read these tweets and reply. We're building a ROBOT that can do this.

---

## The 4 Steps (What Each File Does)

### Step 1: CLASSIFIER (`src/classifier.py`)
**What it does:** Reads a tweet and figures out WHAT it's about.

**Example:**
```
Tweet: "My phone keeps crashing"
Classifier says: "This is a TECHNICAL SUPPORT issue" (95% confident)
```

**How it works:** Uses something called "TF-IDF" which converts words to numbers, then a simple algorithm learns patterns.

---

### Step 2: RETRIEVER (`src/retriever.py`)
**What it does:** Finds SIMILAR old conversations from the database.

**Example:**
```
New tweet: "My iPhone keeps crashing"
Retriever finds 3 similar old tweets:
  1. "Phone restarts randomly" → Apple replied: "Try restarting..."
  2. "Apps keep closing" → Apple replied: "Update your iOS..."
  3. "Screen freezes" → Apple replied: "Visit support.apple.com..."
```

**How it works:** Uses "cosine similarity" to find tweets that are mathematically similar.

---

### Step 3: GENERATOR (`src/generator.py`)
**What it does:** WRITES the actual reply using templates.

**Example:**
```
Input: "My iPhone keeps crashing" + 3 similar old conversations
Output: "We're sorry to hear that! Try restarting your phone. If the issue persists, visit support.apple.com for more troubleshooting steps."
```

**How it works:** Sends the tweet + examples to GPT-3.5 and asks it to write a reply.

---

### Step 4: ESCALATION (`src/escalation.py`)
**What it does:** Decides: Can the robot handle this, or should a HUMAN do it?

**Example:**
```
Tweet: "Where is my order?" → Robot handles (simple question)
Tweet: "I'm going to sue you!" → Escalate to human immediately
Tweet: "I have 3 different problems" → Escalate (too complex)
```

**How it works:** Uses rules like "if tweet contains 'lawyer' or 'sue', escalate."

---

## How to Run This (Step by Step)

### Day 1: Setup (30 minutes)

```bash
# 1. Open terminal/command prompt
# 2. Go to the project folder
cd C:\Users\iamro\OneDrive\Desktop\Infosyys\hiver-project

# 3. Create virtual environment
python -m venv venv

# 4. Activate it (Windows)
venv\Scripts\activate

# 5. Install required packages
pip install pandas numpy scikit-learn openai tiktoken jupyter matplotlib seaborn
```

---

### Day 2: Get the Data (1 hour)

**Option A: Kaggle API (easier)**
```bash
# 1. Install Kaggle
pip install kaggle

# 2. Go to kaggle.com, create account, go to Settings > API > Create New Token
# 3. This downloads kaggle.json file
# 4. Put it in C:\Users\iamro\.kaggle\ folder

# 5. Download dataset
kaggle datasets download -d thoughtvector/customer-support-on-twitter

# 6. Extract it
unzip customer-support-on-twitter.zip -d data/
```

**Option B: Manual Download**
1. Go to https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
2. Click "Download" button
3. Extract the zip file
4. Put `twcs.csv` in the `data/` folder

---

### Day 3: Explore Data (2 hours)

```bash
# Start Jupyter notebook
jupyter notebook

# Open notebooks/exploration.ipynb and run each cell
```

**What you'll learn:**
- How many tweets are in the dataset
- Which brands have the most tweets
- What customer complaints look like
- How conversations are structured

---

### Day 4: Pick Your Brand (1 hour)

You need to pick ONE brand to focus on. Good choices:
- **Apple** - lots of technical support questions
- **Amazon** - lots of order/shipping questions
- **Uber** - lots of complaints

**How to pick:**
1. Look at the data
2. Count how many tweets each brand has
3. Pick one with at least 10,000 tweets

---

### Day 5-7: Build the Agent (3 days)

**Day 5: Intent Discovery**
1. Read 200 random tweets from your brand
2. Group them by topic
3. Write down 8-15 "intents" (categories)

Example intents you might find:
```
- order_status (where is my order?)
- billing_issue (charged wrong amount)
- technical_support (app not working)
- account_access (can't log in)
- refund_request (want money back)
```

**Day 6: Label Training Data**
1. Take 500-1000 tweets
2. Manually label each one with its intent
3. Save as CSV file

**Day 7: Train & Test**
1. Train the classifier on your labeled data
2. Test it on 100 unseen tweets
3. Check accuracy (aim for 70%+)

---

### Day 8-9: Build Evaluation (2 days)

**Day 8: Create Golden Set**
1. Pick 200 test examples
2. For each, write the CORRECT intent
3. For each, write what the ideal reply would be
4. Save as `evaluation/golden_set.json`

**Day 9: Build Eval Script**
1. Run your agent on all 200 examples
2. Compare predictions to golden set
3. Calculate accuracy metrics

---

### Day 10-11: Failure Analysis (2 days)

1. Find 20-30 examples where your agent FAILED
2. For each failure, write:
   - What happened (wrong intent? bad reply?)
   - Why you think it failed
   - How you might fix it

**Common failures:**
- Intent misclassification ("billing" classified as "general")
- Hallucination (made up information)
- Wrong escalation (escalated simple question, or missed complex one)

---

### Day 12-14: Write Report & Submit (3 days)

**Report Structure (6 pages max):**
1. Problem Framing (1 page)
2. Implementation (1.5 pages)
3. Results (1.5 pages)
4. Failure Analysis (1 page)
5. What's Misleading (0.5 page)
6. What I'd Do Next (0.5 page)

**Also write:**
- Decision Log (10-15 decisions you made and why)
- README (how to reproduce your results in 15 min)

---

## Key Concepts Explained Simply

### TF-IDF
Converts text to numbers. "Apple" appears a lot → low importance. "AirPods" appears less → higher importance.

### Cosine Similarity
Measures how similar two pieces of text are. Score of 1.0 = identical, 0.0 = completely different.

### LLM-as-Judge
Instead of humans checking if replies are good, we ask ChatGPT to grade them. Faster, cheaper, scales to thousands.

### Golden Set
Your "answer key" - test questions with known correct answers. Like a school exam.

### Escalation
When the robot says "I can't handle this, send to a human." Important for safety.

---

## What Hiver Cares About

They want to see:

1. **Build → Test → Evaluate → Analyze** (not just build and hope)
2. **Honest failure analysis** (not hiding problems)
3. **Critical thinking** (why your numbers might be misleading)
4. **Practical implementation** (something that actually works)

**They explicitly said:**
> "The proof is worth more than the system."

Meaning: They care more about HOW you prove it works than the system itself.

---

## Common Beginner Mistakes to Avoid

1. **Don't skip the golden set** - This is how you prove your system works
2. **Don't claim 99% accuracy** - It's probably misleading
3. **Don't ignore failures** - They want to see you understand limitations
4. **Don't overcomplicate** - Simple working > complex broken
5. **Don't forget the README** - They need to reproduce your results

---

## Getting Help

If you're stuck:
1. Read the error message carefully
2. Google the error
3. Check if all packages are installed
4. Make sure your API key is set
5. Try running just one file at a time

**Remember:** This project is about LEARNING, not perfection. Show you understand the process.
