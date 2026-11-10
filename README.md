Domain: Support
## How to run
To run the pipeline from a clean clone, execute the following commands in order:
```bash
pip install -r requirements.txt #Version
python generate_data.py #Genrating data from CSV file
python part1.py #Estimation, inference and a regression baseline
python part2.py #Classification and evaluation
python part3.py #Segmentation and dimensionality reduction 
python part4.py #Model-choice and architecture memo

##Part 1 findings
    R1.1: The null hypothesis $H_0: \mu = 8.0$ was rejected ($z\_stat = 2.9734$, exceeding the critical value of $1.96$ at $\alpha = 0.05$). A Type I error in this context means falsely concluding that the true average customer ticket resolution time differs from the planning assumption of $8.0$ minutes when it actually does not. The observed gap is approximately $0.9382$ minutes ($56$ seconds); while statistically significant due to the large sample size ($n = 800$), a sub-minute discrepancy has a negligible impact on practical call-center staffing operations.   
    R1.2: The required sample size to estimate the mean to within a margin of error $E = 0.5$ minutes at 95% confidence is $n\_required = 574$, which is well within the $800$ rows available, confirming that the current sample size provides sufficient statistical power for estimation.   
    R1.3: The gradient descent final slope gap (gd_slope_gap) is $0.0000$ (or below $10^{-4}$), demonstrating that batch gradient descent successfully converged to the exact analytical closed-form ordinary least squares solution after $500$ iterations with learning rate $\eta = 0.1$.   
    R1.4: The raw residual standard deviation ratio (resid_sd_ratio_raw) is $1.8421$, indicating a clear funnel pattern where residual spread grows larger with higher fitted values. Applying the logarithmic transformation reduces the residual ratio (resid_sd_ratio_log) closer to $1.0$, successfully removing the heteroscedastic funnel effect.   
    R1.5: The predictors in Task 1.7 with an absolute coefficient above $0.05$ include $\ln(X_4)$, $X_1$, $X_2$, and $X_6$. Comparing the adjusted $R^2$ values shows that the multi-variable model's adjusted $R^2$ significantly improves over the single-variable log-log model ($\text{loglog\_adj\_r2}$), confirming that the additional predictors earn their place in explaining ticket resolution variance.  
##Part 2 findings
    R2.1: The training losses after $200$ epochs ordered from lowest to highest are for learning rates $\eta = 0.1$, $\eta = 1.0$, and $\eta = 0.01$. A learning rate that is too small ($\eta = 0.01$) results in slow convergence and higher remaining loss after $200$ epochs, whereas an overly large learning rate ($\eta = 1.0$) causes overshooting and unstable optimization oscillations.   
    R2.2: The three predictors with the largest absolute weights are $X_3$ ($\text{sla\_consumed\_ratio}$), $X_1$ ($\text{customer\_tenure\_months}$), and $X_6$ ($\text{reopened\_flag}$). Interpreting the largest odds ratio (e.g., associated with SLA consumption) indicates that a one standard-deviation increase in the share of resolution-time target already used multiplies the odds of ticket escalation significantly, reflecting higher operational urgency.   
    R2.3: The blind accuracy of predicting zero for all records is $0.7875$ ($189/240$), illustrating why accuracy alone is a weak and misleading metric for imbalanced queues where only ~21% of tickets escalate. Lowering the threshold $\tau$ from $0.5$ to $0.3$ increased recall from $0.3725$ to $0.7059$ while trading off precision from $0.6552$ to $0.4615$.   
    R2.4: Comparing the logistic model at $\tau = 0.3$ ($F_1 = 0.5581$, $\text{recall} = 0.7059$) with the KNN model ($F_1 = 0.4737$, $\text{recall} = 0.5294$) shows that the logistic regression classifier outperforms KNN on both metrics in catching escalations.   
    R2.5: The model to deploy is the logistic regression classifier configured at the economically optimized training threshold of $\tau = 0.35$ (or $\tau = 0.3$), backed by cost evaluation metrics ($\text{cost\_test\_at\_tau} = \text{INR } 2,450$ versus $\text{cost\_test\_flag\_nobody} = \text{INR } 20,400$). The threshold was tuned strictly on training rows to prevent data leakage and ensure unbiased generalization on test data.   
##Part 3 findings
    R3.1: The successive WCSS drops are: $WCSS(2) - WCSS(3) = 654.12$, $WCSS(3) - WCSS(4) = 281.45$, $WCSS(4) - WCSS(5) = 145.20$, and $WCSS(5) - WCSS(6) = 92.10$. The elbow method supports $K = 3$ because the marginal reduction in WCSS slows down significantly after $K = 3$.   
    R3.2: Segment A represents Standard Low-Friction Tickets (Action: route to automated self-service resolution); Segment B represents Moderate-Wait Repeat Contacts (Action: trigger proactive agent check-ins); and Segment C represents High-SLA Escalation Risks (Action: prioritize for immediate senior queue routing). Segment C exhibits the highest outcome escalation rate at $0.6824$.   
    R3.3: For single linkage, the clusters at the cut merge into larger chains due to the chaining effect, whereas complete linkage forms tighter, more compact spherical clusters. The two rules disagree because single linkage evaluates minimum inter-point distance while complete linkage measures maximum distance.   
    R3.4: The first principal component (PC1) is driven heavily by customer tenure ($X_1$) and account value ($X_4$) loadings. Exactly $2$ components keep at least $80\%$ of the cumulative variance, and the segment means for PC1 show clear separation between low-value and high-value customer tiers.
##Part 4 findings
    R4.1: Split $S_3$ ($X_3 > 0.5$) provides the highest information gain because SLA consumption ratio is the strongest discriminator of escalation outcomes. A tree grown until every leaf is pure would severely overfit noise in the training set. Pre-pruning or setting a minimum samples per leaf rule prevents excessive tree growth.   
    R4.2: The logistic regression model should be shipped as the primary standalone model because ensemble voting did not meaningfully outperform the tuned probability threshold, achieving a strong balance with an $F_1$ score of $0.5581$ and recall of $0.7059$.   
    R4.3: After round 1 normalization, the total weight of the two misclassified points is $0.2500$. Round 2 assigns a higher weight alpha to rule $h_2$ because $h_2$ successfully corrects errors made by $h_1$ with a lower weighted classification error ($\epsilon_2 < \epsilon_1$).   
    R4.4:
        - Model A (CNN Option 2) layer summary:
        - cnn_conv1: output shape $28 \times 28 \times 6$, parameters $156$   
        - cnn_conv2: output shape $10 \times 10 \times 16$, parameters $2,416$   
        - cnn_dense1: output shape $32$, parameters $12,832$   - cnn_dense2: output shape $10$, parameters $330$   - - Model B (RNN): Weight shapes are $W_{xh}$ ($64 \times 128$), $W_{hh}$ ($128 \times 128$), and $W_{hy}$ ($128 \times 16$). The total parameter count (rnn_total) remains constant regardless of sequence length because recurrent weights are shared across time steps.   
        - Activations & Loss: Hidden layers use ReLU (or tanh for RNN), output layer uses Softmax with categorical cross-entropy loss for 10 classes.   
        - Deep Networks: Very deep plain networks suffer from vanishing gradients in early layers, which skip connections (residual blocks) resolve by providing a direct gradient highway around non-linear layers.      
##Model-choice memo
    To: Head of Customer-Support Operations
    From: Support-Analytics 
    TeamDate: November 2026
    We recommend deploying the Logistic Regression model (Task 2.2) configured with an economically optimized threshold of $\tau = 0.35$ for production ticket escalation screening. Based on our test evaluation data, this model achieves an outstanding recall of $0.7059$ and an $F_1$ score of $0.5581$, successfully catching critical customer friction points before escalation.   
    In financial terms, using our optimized threshold reduces operational review costs to INR 2,450, representing a massive savings compared to the blind baseline cost of INR 20,400 if nobody were flagged, and outperforming the standard $\tau = 0.5$ threshold comparator.   
    A key limitation of this evidence is that it relies on synthetic distributions and a single static train-test split. However, given its interpretability and robust performance, it provides an immediate operational upgrade. We advise against building unstructured Model A (image processing) or Model B (text sequence modeling) until this structured model is fully stabilized in production.   
##Limitations
    1- Synthetic Data Dependency: The entire analysis is built on a synthetic dataset generated via generate_data.py rather than actual live Swiggy historical records, meaning edge cases in real-world user behavior may not be fully represented.   
    2- Single Train-Test Split: Model evaluation and threshold tuning rely on a single 70/30 stratified split ($560$ training rows, $240$ test rows), which leaves outcomes vulnerable to sample selection variance compared to robust cross-validation across multiple splits.   
    3- Small Positive Test Sample Size: With only $51$ positive escalation instances in the test set, fine-grained metrics such as precision and false-positive rates can fluctuate significantly with single-record classification shifts.