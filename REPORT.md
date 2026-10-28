# Assignment 2 Report

- **Name**: Maria Fernandez Gomez-Monche
- **Student ID**: 19784
- **Email**: mfernandezg.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

# Business problem, and the cost of being wrong
I framed this as an early-warning problem. Cameras watching a fire prone area send a photograph every 10 minutes, and for each one of them, the system can pass it, fail it (smoke, so an alert goes out asking for help) or send it to an operator in the control room, who looks and decides. Smoke is my positive class because it usually appears before flames. It's also the weaker class in my data, so I wanted the headline number to be about smoke and not an average over both classes.

The two mistakes don't cost the same. A false alarm costs a wasted patrol trip. Missing smoke lets a small fire grow, and the few fires that escape cause most of the damage: fewer than 2% of fires account for 95% of the area burned and 85% of of the money spent on suppression (US Forest Service, https://www.fs.usda.gov/rm/pubs_journals/2016/rmrs_2016_katuwal_h001.pdf).

All the euro amounts are my assumptions. A missed smoke costs €5000: a 5% chance of escape times €100000 for an escaped fire. US data says 96 to 98% of fires are controlled in the first burning period, so I put the 5% a little above the normal 2 to 4% since a missed detection is late. The €100000 is my estimate. If a missed smoke were worth less than about €2000, doing nothing would beat the model. A false alarm costs €150 (2 people for 1.5 hours at €35 an hour, plus €45 for the vehicle), and above about €380 the model would lose to doing nothing. A review costs €0.50 (about 30 seconds of operator time), and I assume the reviewer misses 5% of real smoke and wrongly confirms 2% of clean images. Correct alerts and passes cost nothing extra.

The assumption I'm least sure of is the base rate. I used 0.1% of snapshots containing smoke, a guess that is nothing like the 37% in my test set. Below about 0.05%, doing nothing would be cheaper than the model.

# Dataset
I used the Roboflow Universe dataset `sayed-gamall` / `fire-smoke-detection-yolov11`, version 2 (https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11). An image is a camera snapshot and the labels are boxes around visible flames (Fire) and smoke plumes (Smoke). It mixes CCTV, industrial, residential and drone images instead of tower cameras, its smoke is large (median 16% of the image) while early smoke is small and distant, and it has no ordinary scenes. It can support a model that finds smoke, but not a claim about early detection from far away.

# What the dataset's composition implies
The splits are 9,156 images for train, 872 for validation and 435 for test. Fire boxes outnumber smoke boxes about 2.7 to 1 in train (11,505 against 4,312), and the test split has only 216 smoke boxes, so my most important number rests on my smallest sample. Train is also augmented: its 9,156 images come from 2,419 originals.

The bigger problem is overlap between splits. I compared the actual pixels, mirrored copies included and 294 of the 435 test images (68%) are near-duplicates of a train or validation image. My check misses crops and rotations, so the real figure is probably higher. That makes the validation score I saw while training (about 0.77 mAP50) optimistic and the full test score too. So I also report a clean test set of the 141 test images with no near-duplicate (91 with smoke, 50 without). It's the number I trust, but it's small.

Every test image also contains fire or smoke and the 274 images without smoke all contain fire. My negatives are "fire without smoke", not "a normal day", so the false-alarm rate on real footage is unknown. I can't conclude that the model works on early or distant smoke or on new cameras.

# Finetuning and baselines
I used object detection, since the data has boxes and my decision needs the best smoke box score per image. I finetuned yolo11s from the pretrained COCO checkpoint, choosing s over n for a bit more capacity on small, faint smoke (I didn't compare them). I trained on Google Colab with a T4 GPU for about 2 hours: 50 epochs, image size 640, batch 16, patience 15, seed 0. The weights are in `weights/best.pt`, picked on the leaky validation split, from one run. To reproduce it, put your Roboflow key in `.env` and run `uv run python main.py train`.

On the same test split, at confidence 0.25, the finetuned model catches 82% of smoke images on the clean set with 96% precision. The pretrained COCO model finds nothing relevant, because COCO has no fire or smoke class. Predicting nothing misses all 161 smoke images, and always alerting catches them all but is right only 37% of the time. The finetuned model beats all three.

# Per-class evaluation and failure modes
Smoke carries the headline number: on the clean test set its AP50 is 0.73 (mAP50-95 0.38), with precision 0.87 and recall 0.62. Fire gets an AP50 of 0.69 there but 0.85 on the full test set, probably leakage helping it. Smoke barely changes (0.71 on the full set), though with 91 images I can't say it's immune.

At confidence 0.25 on the clean set, the model finds 72 of 109 smoke boxes, misses 35 and invents 13 on empty background. It almost never confuses the two classes. So smoke recall, around two thirds, is the weak spot, and a missed smoke is my expensive mistake. My decision also ignores where the box is, so a smoke box in the wrong place still counts as correct.

# Cost matrix and operating point
Each image's score is its best smoke confidence, and an alert fires when it reaches the threshold. I measured the true- and false-positive rates at every threshold on the clean test set, then reweighted them to the assumed 0.1% base rate, because the raw 37% would overweight misses.

Per 1000 cases, doing nothing costs €5000. The model alone at the default 0.25 costs €11,181, worse than doing nothing, because even a 6% false-alarm rate across about 999 clean cases costs roughly €10000. At its best threshold, 0.53, it costs €3290. Model plus human review, passing below 0.03 and failing from 0.53, costs €2306 and that's my operating point. The 0.53 is fragile, though: it sits where false alarms vanish in just 50 clean negatives, and resampling the test images moves it between 0.32 and 0.53.

The base rate is what would change my answer most. At 0.03% doing nothing is still cheaper (€1,500 against €1,884). At 0.1% model plus review clearly wins, and at 1% it costs €4,270 against €50000. The break-even is about 0.05%.

# What a person still has to check
I assumed 40 cameras sending a snapshot every 10 minutes, so 5760 a day. My thresholds send everything scoring between 0.03 and 0.53 to a person: 17.7% of cases, about 1,000 reviews a day, or 8.5 hours of work at 30 seconds each. With one reviewer per shift on three 8-hour shifts, spending 75% of their time reviewing, that uses 47% of the capacity and needs a roster of about 5 people. I think that's realistic.

Against the alternatives, per 1000 cases: doing nothing costs €5000; having a person check everything costs €3747 but needs about 12 people when 5 are available; letting the model decide alone costs €3290 and misses about 36% of smoke; model plus review costs €2,306, which is 54% cheaper than doing nothing. The weak point is false alerts, about 77 a day against about 5.5 real ones, so a person would need to confirm every alert before a patrol goes out.

# Limitations & next steps
The main limitation is the leakage plus a small clean test set. With more time I'd re-split by source image or camera before training and collect enough new test images that smoke has several hundred boxes. I'd add thousands of empty and hard-negative frames (fog, cloud, dust, steam) so false alarms are measured on ordinary scenes, and test on a tower-camera dataset such as `sjmon/wildfire-smoke-v8d0m` . A fire with no smoke currently passes, because my decision is about smoke only, so I'd give fire its own rule. My cost figures are all assumptions, so I'd get real ones from the agency. I also treated snapshots as independent although a plume stays in view for many frames, so I'd evaluate at the event level. And I trained once, so I'd rerun on a clean split with several seeds.

# Generative AI use disclosure
I used Claude (Anthropic's AI assistant) throughout this assignment. I discussed the problem and datasets with it, and it searched for the datasets and wildfire figures I cite. It wrote the first versions of `train.py`, `costing.py` and `app.py` and fixed bugs as I ran them. I ran everything myself, so all the numbers come from my own runs.