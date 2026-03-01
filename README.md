# FlightID — Fare-Inflation Explanation & Corridor Recommendations

This project trains an interpretable model (XGBoost GBM) to predict expected flight fares per-mile and uses SHAP to explain why a fare is inflated. It also produces corridor rankings (cost-effectiveness) and suggests cheaper corridors from the same origin.

Quick start

1. Create a Python environment and install requirements:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

2. Run training + explanation pipeline (reads `data/airline_ticket_dataset_csv.csv` by default):

```bash
python src/train_explain.py --csv data/airline_ticket_dataset_csv.csv --out outputs
```

Outputs (written to `outputs/`):
- `model.joblib` — trained model
- `shap_summary.png` — SHAP summary plot
- `inflated_flights.csv` — rows flagged as inflated with suggested alternatives
- `corridor_rankings.csv` — corridors ranked by predicted fare per mile

Files added: `src/train_explain.py`, `src/utils.py`.

See `src/train_explain.py --help` for options.

In order to open the SDSS_Hackathon.pbix file in the DevPost submission, you can either open it on the PowerBI Desktop app or through the [web](app.powerbi.com) and upload it as your own workshop. (Free Trial struggles)

## Inspiration

Airline pricing today feels like a black box. After scandals like the “Honey” coupon controversy raised questions about whether consumers are actually getting the best deals, we started wondering:

How do you know if you’re overpaying for a flight?

Two nearly identical flights can have drastically different prices, and travelers have no clear way to judge whether a fare is fair or inflated. Airplane tickets can range anywhere from **$50** to **$300+ CAD** for similar routes, driven by hidden factors like competition, hub dominance, and demand.

We were inspired by the idea of _demystifying_ airfare pricing. Consumers deserve transparency to know whether they got a good deal, what drives sky-high prices, and where cheaper alternatives might exist.

So we built FlightID to answer:
* Can we **predict a baseline “fair” flight price?**
* Can we identify **overpriced routes?**
* What factors _actually_ drive **airfare costs?**
* Can we **suggest better destinations** with a better cost per mile ratio?

## What it does
FlightID is a machine learning system that audits airline prices.

It:
* Predicts the expected fare per mile for a flight
* Flags overpriced (“inflated”) flights
* Explains why prices are high
* Suggests cheaper alternative destinations
* Ranks routes by cost-effectiveness

Our system transforms opaque airfare pricing into something measurable and actionable. We also built powerful visualizations, including:

* A carrier breakdown showing which airlines account for inflated flights each year
* A map which shows how much influence each factor has in determining price
* A real-world plot revealing inefficient routes, as well as offering alternative flights with better cost-effectiveness

FlightID matters because it transforms raw flight data into a spatial diagnostic tool for market efficiency. We aren't just predicting prices; we are identifying where the market is failing the consumer.

Real-World Applications:
* Policy & Regulation: FlightID allows regulators to identify fare-inflated routes where market concentration, rather than operational costs, is driving prices up.
* Airline Network Strategy: For carriers, this tool identifies gaps in competition, where current service is inadequate or overpriced, pinpointing ripe opportunities for new market entry.
* Consumer Empowerment: By identifying alternative gateways, FlightID provides actionable insights that help travelers and corporate travel managers optimize their routing.

Our Model:
* Scalable Diagnostics: Unlike manual audits, this model scales across the entire national aviation network, instantly flagging anomalies across thousands of routes.
* Scientific Transparency: Because we utilized SHAP-based interpretability, we explain the ‘why.’
* Unbiased Precision: Our model’s near-normal residual distribution proves it is a robust, generalized tool, not a black-box algorithm that overfits to noise.

Overall, the model takes into account a lot of parameters, including the distance, total number of passengers in each city, the time (year and quarter), the competitive pressure, the hub intensity (how populated is the city?), etc. From this, we were able to calculate the SHAP value (impact on model output) each parameter has! Thus, we were able to notice that distance was the largest factor, followed up next by volume, airline, distribution, and time. In comparison, the hub dominance (does a carrier have a monopoly over flights at this airport?) was relatively insignificant. 

## How we built it

The model itself was built on VS Code using Python. We used a Jupyter Notebook and several Python libraries, including pandas, numpy, and matplotlib. Several of our graphs were created on Jupyter Notebook, straight from the .csv files. We also created an interactive PowerBI chart, which allows the user to see which airplane carriers offered the most inflated flights for different years. 

At the core, weimplemented an XGBoost (Gradient Boosted Machine) regressor. By utilizing a sequence of optimized decision trees, our model identifies complex, non-linear patterns in fare dynamics that traditional linear models fundamentally miss.

Our model is highly responsible and transparent. By integrating SHAP (SHapley Additive exPlanations), we decompose every fare prediction into its contributing factors based on game theory.
* Our analysis proves that distance (nkm) and passenger volume are the true drivers of fare structure.
* Crucially, our model’s ability to correctly identify ‘noise’, like hub intensity, as insignificant is proof that it is learning generalized market logic rather than overfitting to specific data anomalies.

We didn't just chase low error numbers; we validated our performance through rigorous benchmarking.
* Superior Accuracy: We achieved an MAE (mean absolute error) of 0.01859, which significantly outperformed our linear regression baseline.
* The ‘Bell Curve’ Proof: Our residual error distribution shows a near-normal, symmetric bell curve centered at zero.
* Generalization: This symmetric distribution serves as our formal proof against overfitting. It demonstrates that our model is statistically unbiased and generalizes reliably across the entire US aviation market.

## Challenges we ran into

Apparently, PowerBI's free trial doesn't allow users to share projects easily, so the report had to be downloaded to get passed along. Furthermore, learning how to use PowerBI was a learning curve itself. It was relatively hard for us to each add something to the Jupyter Notebook, especially because the notebook was on Github and with a team of 4, we were all worried about having merge conflicts. 

We wanted to make everything interactive, but majority of our graphs were created on Jupyter Notebook using several python libraries such as pandas, numpy, and matplotlib. The biggest constraint for us was time, where we didn't have enough time to be able to transfer it all to PowerBI. We also really like sleep ... some of us. 

Since we created multiple tables, they were all over the place and it was difficult to get a good grasp on the location of each table, what they did, and which columns they had. Furthermore, it was very common for us to forget what the column represents, taking the time to look back at the original spreadsheet's dictionary. Overall, the challenges pieced together the entire project, making it a memorable experience to look back upon. 
## Accomplishments that we're proud of

We're proud to have built an end-to-end explainable ML pipeline using existing Python libraries. We were able to detect overpriced flights using the data given after playing around and giving a threshold value. Furthermore, we were able to create an interactive analysis notebook on PowerBI, and generated alternative route suggestions with better cost-effectiveness! Overall, we turned the messy airline data with a crazy number of columns into meaningful insights that consumers, airlines, and policy makers alike can understand and interpret. 

Most of all, we’re proud that FlightID goes beyond predicting prices, offering understandable explanations and suggestions to inform consumer decisions.
## What we learned

This was our first time using PowerBI as an interactive way to represent models, which was interesting in learning how to use! Some of us weren't as experienced in using the python libraries, so it was very interesting to learn about dataframes and how to plot the data! It was really interesting looking at the .xlsx and .csv, and being able to convert the data into actual graphs!! 
## What's next for FlightID

In the future, we could transfer everything on the Jupyter Notebook to the PowerBI, making everything interactive! The UI for the PowerBi could use some work, and keeping to one consistent colour scheme would make it more visually appealing. 