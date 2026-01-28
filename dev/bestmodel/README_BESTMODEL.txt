BEST MODEL TRAINING - STEP BY STEP
==================================

You will run 2 scripts. Follow these steps exactly.


STEP 1: OPEN TERMINAL
---------------------
In VSCode: View > Terminal (or press Ctrl+`)

Then navigate to the project folder. Type this and press Enter:

    cd path/to/PRL_team_11

(Replace "path/to" with wherever you have the repo cloned)


STEP 2: RUN BESTMODEL SCRIPT (7 seeds)
--------------------------------------
First, open this file in VSCode:
    dev/bestmodel/run_bestmodel.sh

Edit the parameters at the top (lines 10-14):
    GAMMA=0.9              <-- change to best gamma from sweep
    ALPHA=0.1              <-- change to best learning rate from sweep
    EPISODES=160           <-- change to best episode count from sweep
    REWARD_SHAPING=true    <-- keep true or change to false
    EPSILON_DECAY=0.995    <-- change if needed

Save the file.

Then in terminal, run:

    bash dev/bestmodel/run_bestmodel.sh

Wait for it to finish. This trains 7 models (seeds 4-10).


STEP 3: RUN BEST CHECKPOINTS SCRIPT (all 10 seeds)
--------------------------------------------------
First, find your 3 sweep run folders. Look in:
    results/qlearning/

Find the 3 folders from your sweep that have the best parameters.
Copy their folder names (e.g., "20250128_143022_s50.0_seed1").

Open this file in VSCode:
    dev/bestmodel/run_best_checkpoints.sh

Edit lines 11-13:
    SWEEP_RUN_1="paste_folder_name_here"
    SWEEP_RUN_2="paste_folder_name_here"
    SWEEP_RUN_3="paste_folder_name_here"

Save the file.

Then in terminal, run:

    bash dev/bestmodel/run_best_checkpoints.sh

Wait for it to finish. This saves the best Q-tables for all 10 seeds.


STEP 4: CHECK OUTPUTS
---------------------
Your Q-tables for submission are in:
    results/best_checkpoints/

Each seed has:
    - bestmodel_seed1_qtable.npz    <-- the Q-table file for submission
    - bestmodel_seed1_validation_rollout.png
    - bestmodel_seed1_config.json


STEP 5: GENERATE SUMMARY (optional)
-----------------------------------
To see results summary, run:

    python3 dev/bestmodel/summarize_results_bestmodel.py

This creates:
    - dev/bestmodel/bestmodel_summary.txt
    - dev/bestmodel/bestmodel_robustness.png


TROUBLESHOOTING
---------------
If you get "permission denied":
    chmod +x dev/bestmodel/run_bestmodel.sh
    chmod +x dev/bestmodel/run_best_checkpoints.sh

If you get "python3 not found":
    Try using "python" instead of "python3"

If you get module errors:
    pip install numpy pandas matplotlib
