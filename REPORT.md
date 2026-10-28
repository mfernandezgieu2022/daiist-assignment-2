# Assignment 2 Report

- **Name**: Maria Fernandez Gomez-Monche
- **Student ID**: 19784
- **Email**: [fill in]
- **Group**: [fill in: BBADBA 5A | BBADBA 5B | PPLEDBA 5A | BDBA 3A]

## Business problem, and the cost of being wrong

Picture a network of fixed cameras watching a fire-prone area, each sending a snapshot every 10 minutes. For every snapshot the system has to decide: pass it, fail it (smoke, so raise an alert and send a patrol of two people), or send it to a person for review. The operators in the agency's control room act on the review cases, and the patrol crews act on the alerts.

My positive class is smoke, because it usually appears before flames and so gives the earliest warning. It is also the weaker class in my data, so the headline number has to be about smoke and not an average that hides it.

The two mistakes are not equal. If we miss smoke, a small fire can grow unnoticed. US data suggests most fires are controlled early, but the few that escape do most of the damage: fewer than 2% of fires account for 95% of the area burned and 85% of suppression spending (Donovan & Brown 2005, cited in https://www.fs.usda.gov/rm/pubs_journals/2016/rmrs_2016_katuwal_h001.pdf). A false alarm only costs a wasted patrol trip.

All my euro figures are assumptions. A missed smoke costs €5,000: a 5% chance it escapes, times €100,000 for the cost of an escaped fire. The 5% is my judgement: US data says 96 to 98% of fires are controlled in the first burning period (https://dl.uc.pt/rest/bitstreams/11169995/retrieve), so 2 to 4% escape, and a missed detection is late by definition. The €100,000 has no source. If a missed smoke were worth less than about €2,000, doing nothing would beat the model. A false alarm costs €150 (two staff for 1.5 hours at €35 an hour, plus €45 for the vehicle), and above roughly €380 the model would lose to doing nothing. A human check costs €0.50, about 30 seconds of operator time. I assume the reviewer misses 5% of real smoke and wrongly confirms 2% of clean images. Correct alerts and correct passes cost nothing extra.

The assumption that worries me most is the base rate. I assume 0.1% of snapshots contain smoke. That's a guess, and it is very different from the 37% in my test set. If the real rate were below about 0.05%, doing nothing would be cheaper than the model.

Everything later follows from this: smoke as the positive class, a threshold chosen by cost instead of the default 0.25, and a review band limited by how many reviews people can actually do.

## Dataset

I used the public Roboflow Universe dataset `sayed-gamall` / `fire-smoke-detection-yolov11`, **version 2** (https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11), CC BY 4.0. I didn't fork it. One image is a camera snapshot, and the labels are boxes around visible flames (Fire) and smoke plumes (Smoke).

It fits my problem only partly. It gives me thousands of real boxed images of both classes, but it mixes CCTV, industrial, residential and drone pictures instead of tower cameras, its smoke is large (a median of 16% of the image) while early smoke is small and far away, and it has no ordinary scenes. So it can support a model that finds smoke, but not a claim about early detection from a distance.

## What the dataset's composition implies

The splits are 9,156 images for train, 872 for validation and 435 for test. Fire boxes outnumber smoke boxes about 2.7 to 1 in train (11,505 against 4,312), and the test split has only 216 smoke boxes. Smoke is also my positive class, so my most important number rests on my smallest sample. Train is also augmented: its 9,156 images come from only 2,419 originals.

The bigger problem is that the splits overlap. I compared the pixels of the images, mirrored copies included. Of the 435 test images, 294 (68%) are near-duplicates of a train or validation image, and the validation split has the same problem (64% overlap with train). This is a lower bound, since my check misses crops and rotations. The validation score I saw during training (about 0.77 mAP50) is therefore optimistic, and so is the full test score. So I also report a clean test set: the 141 test images with no near-duplicate (91 with smoke, 50 without). It is honest but small, and I wouldn't trust small differences in its smoke numbers.

A second problem is that every test image contains fire or smoke, and the 274 images without smoke all contain fire. My negatives are "fire without smoke", not "a normal day", so the false-alarm rate on real footage is unknown. I can't conclude that the model works on early or distant smoke, or on new cameras.

## Finetuning and baselines

I used object detection, since the data has boxes and the decision needs the score of the best smoke box in each image. I finetuned yolo11s from the pretrained COCO checkpoint. I picked s over n for a bit more capacity on small, low-contrast smoke, and I didn't compare the two. I trained on Google Colab with a T4 GPU, for about 2 hours, with 50 epochs, image size 640, batch 16, patience 15 and seed 0. The weights are in `weights/best.pt`, chosen on the validation split (which leaks), from a single run. To reproduce it, put your Roboflow key in `.env` and run `uv run python main.py train`; `SKIP_TRAIN=1` reuses the saved weights.

On the same test split, the finetuned model catches about 82% of smoke images on the clean set with 96% precision (at confidence 0.25). The pretrained COCO model finds nothing relevant, as expected, because COCO has no fire or smoke class. A model that predicts nothing misses all 161 smoke images, and one that always alerts catches them all but is right only 37% of the time. The finetuned model beats all three, by about as much as I expected.

## Per-class evaluation and failure modes

Smoke carries the headline number. On the clean test set its AP50 is 0.73 (mAP50-95 0.38), with precision 0.87 and recall 0.62. Fire scores an AP50 of 0.69, and on the full test set fire jumps to 0.85, which is probably leakage helping it. Smoke barely changes (0.71 on the full set), though with 91 images I can't say it's immune.

The confusion matrix shows where it breaks. At confidence 0.25 on the clean set the model finds 72 of 109 smoke boxes and misses 35, and it invents 13 on empty background. It almost never mixes up the two classes (2 smoke boxes called fire). It mostly misses things or invents them. So smoke recall, around two thirds, is the weak spot, and a missed smoke is my expensive mistake. One more caveat: my decision ignores where the box is, so a smoke box in the wrong place still counts.

## Cost matrix and operating point

For each image the score is its best smoke confidence, and an alert fires when it reaches the threshold. I measured the true- and false-positive rates at every threshold on the clean test set and reweighted them to the assumed 0.1% base rate, because the raw 37% would overweight misses.

Per 1,000 cases, doing nothing costs €5,000. The model alone at the default 0.25 costs €11,181, worse than doing nothing, because even a 6% false-alarm rate across the roughly 999 clean cases costs about €10,000. At its best threshold, 0.53, it costs €3,290. Model plus human review, passing below 0.03 and failing from 0.53, costs €2,306, and that is my operating point. Picking a threshold by taste would have lost money. The 0.53 is fragile, though: it sits where false alarms vanish in only 50 clean negatives, and when I resample the test images it moves between 0.32 and 0.53.

The assumption that would change my answer most is the base rate. At 0.03% doing nothing is still cheaper (€1,500 against €1,884). At my 0.1% the model plus review wins clearly, and at 1% it costs €4,270 against €50,000. The break-even is around 0.05%.

## What a person still has to check

With 40 cameras sending a snapshot every 10 minutes (5,760 a day), my thresholds send everything scoring between 0.03 and 0.53 to a person. That's 17.7% of cases, about 1,000 reviews a day, or 8.5 hours of work at 30 seconds each. I assumed one reviewer per shift on three 8-hour shifts, spending 75% of their time reviewing, so the plan uses 47% of that capacity and needs a roster of about 5 people. I think that's realistic.

Compared with the alternatives (per 1,000 cases): doing nothing costs €5,000. Having a person check everything costs €3,747 but would need about 12 people against 5 available. Letting the model decide alone costs €3,290 and misses about 36% of smoke. Model plus review costs €2,306, which is 54% cheaper than doing nothing.

The weak point is false alerts: about 77 a day against about 5.5 real ones. The model shouldn't dispatch patrols by itself. A person would have to confirm every alert first.

## Limitations & next steps

The main limitation is leakage plus a small clean test set. With more time I'd re-split by source image or camera before training and collect enough new test images that smoke has several hundred boxes. I'd also add thousands of empty and hard-negative frames (fog, cloud, dust, steam) so false alarms are measured on ordinary scenes, and test on a tower-camera dataset such as `sjmon/wildfire-smoke-v8d0m` (I haven't checked its classes).

A visible fire with no smoke currently passes, because my decision is about smoke only, and the dashboard warns about it. I'd give fire its own rule. My cost figures are all assumptions, so I'd get real ones from the agency and time how long reviewers actually take. I also treated snapshots as independent, although a plume stays in view for many frames, so I'd evaluate at the event level and require several snapshots to agree. Finally, I trained once with one seed, so I'd rerun it on a clean split with several seeds.

## Generative AI use disclosure

I used Claude (Anthropic's AI assistant) throughout this assignment. [Edit this to match exactly what you did and what your syllabus requires.] I discussed the problem and dataset options with it, and it searched for the datasets and the wildfire figures I cite. It wrote the first versions of `train.py`, `costing.py` and `app.py` and fixed bugs as I ran them. I ran everything myself, so all the numbers come from my own runs. It suggested the starting cost values [say which you kept or changed and why] and drafted this report [say what you edited or checked yourself].