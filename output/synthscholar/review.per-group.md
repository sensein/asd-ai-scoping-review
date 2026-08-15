# Per-Behavioural Modality (Approximate, Derived From Charted Data Type) Analysis — To characterise how computation-based (machine learning and other computational) autism prediction studies that use behavioral data are designed, conducted and reported: who is studied and how they are described, in what settings and with what tasks, hardware and software the behavioral data are collected, how datasets are shared and how reproducibility and bias are handled, which behavioral modalities are operationalised and how they are combined, which computational methods, representations, evaluation metrics, interpretability techniques and validation strategies are used, and how the publication landscape has evolved over time.

### 3.8 Per-Behavioural Modality (Approximate, Derived From Charted Data Type) Analysis

*9 groups · 170 included articles · 0 unlabeled (excluded from analysis).*

**Overall caveats:** Groups are loosely defined families, not exhaustive mutually exclusive categories; several studies combine modalities and are assigned to the family matching their charted primary data type. Group-level statements describe tendencies and name exemplars rather than asserting properties of every member.


#### Eye tracking and gaze *(n = 49)*

Largest group in the corpus. Reported accuracy is bimodal and tracks the analysis unit: studies partitioning gaze samples, scanpath images or eye crops report 97-100%, while the two studies using participant-independent partitions report F1 0.641 and 0.96 with a fold standard deviation of 0.08 on 28 people. Where interpretability is done properly the group produces its most valuable output - falsifiable claims about which gaze parameters carry information.

*Representative studies:* PMID:local_863980f8, PMID:local_40cd7d53, PMID:local_54954c19, PMID:local_88ddfdd1, PMID:local_96c87d36

*Caveats:* Modality assignment is approximate, derived from each record's charted data-type field rather than from a categorisation applied by the original authors.

**Group-specific Q&A:**

- **Q:** What determines reported performance in this group?
  **A:** The unit of analysis, not the paradigm. local_7432ffd7 split 1,048,575 gaze samples from 59 children by sample (99.78%, with CARS additionally in the feature vector); local_44d3aef5 split 547 scanpath images by image and reported 100% on 55 test images; local_bb4d1f7a generated exactly 100 eye crops per child from 40 children (97%). local_5f57ba19 used participant-independent nested cross-validation with a majority-class baseline built in (F1 0.641) and local_d7b6fa00 aggregated to the participant and listed leakage prevention as a contribution (0.96 ± 0.08 on 28 people).
  *Sources:* PMID:local_863980f8, PMID:local_40cd7d53, PMID:local_54954c19, PMID:local_88ddfdd1, PMID:local_96c87d36
- **Q:** What behavioural findings does the group produce?
  **A:** Where feature analysis is performed: fixation duration and vertical gaze distribution discriminate severity while velocity does not (local_bb4d1f7a); broad social/non-social areas of interest outperform fine decomposition by facial region and emotion (local_b6ac2b98); mouth fixation time and cross-AOI revisits are the leading discriminators, with the audio-visual advantage concentrated before speech onset (local_5f57ba19); mean duration of visits to objects distinguishes groups, read as difficulty disengaging attention (local_b6ac2b98).
  *Sources:* PMID:local_863980f8, PMID:local_40cd7d53, PMID:local_54954c19, PMID:local_88ddfdd1, PMID:local_96c87d36
- **Q:** How is the comparison group defined?
  **A:** Almost always typically developing children. The one study including a second clinical group (local_b6ac2b98, developmental language disorder) found F1 falling from 0.86 to 0.63, and concluded that the approach is more effective as a screening tool than for differential diagnosis.
  *Sources:* PMID:local_863980f8, PMID:local_40cd7d53, PMID:local_54954c19, PMID:local_88ddfdd1, PMID:local_96c87d36


#### Movement, kinematics and interaction traces *(n = 31)*

Contains the corpus's strongest differential-diagnosis study and its most honest handling of a perfect result, alongside frame- and session-level leakage of the same kind found elsewhere.

*Representative studies:* PMID:local_0dbc5018, PMID:local_dc5fc4bb, PMID:local_e404a2a3, PMID:local_780141fc, PMID:local_3a331973

*Caveats:* Modality assignment is approximate, derived from each record's charted data-type field rather than from a categorisation applied by the original authors.

**Group-specific Q&A:**

- **Q:** What does direct comparison with another movement disorder show?
  **A:** local_69b10fe4 compared autistic and parkinsonian movement with matched controls and found no kinematic feature shared between the two conditions that also differed from controls, refuting the anecdotal resemblance; classification reached mean test accuracy 0.73 for clinical versus non-clinical and 0.93 for ASD versus Parkinson's on test partitions of roughly 13 to 19 people.
  *Sources:* PMID:local_0dbc5018, PMID:local_dc5fc4bb, PMID:local_e404a2a3, PMID:local_780141fc, PMID:local_3a331973
- **Q:** How is a perfect result handled?
  **A:** local_231757c7 reported 100% for its severe-ASD class and stated in the same abstract sentence that the class's features were synthesised by averaging moderate-ASD vectors and adding Gaussian noise, computed the 95% interval (39.8-100.0%), noted that one misclassification would reduce recall to 75%, and set a minimum sample of 20-50 real cases for validation. It also reports that Kinect could not be deployed at all for children with severe autism.
  *Sources:* PMID:local_0dbc5018, PMID:local_dc5fc4bb, PMID:local_e404a2a3, PMID:local_780141fc, PMID:local_3a331973
- **Q:** What leakage patterns appear?
  **A:** local_b26db2f2 split 640 game sessions from 12 children by session and reported 96%, with leave-one-child-out available and unused; local_e771fb69 evaluated a four-modality federated framework on nine test cases with class-conditional mean imputation.
  *Sources:* PMID:local_0dbc5018, PMID:local_dc5fc4bb, PMID:local_e404a2a3, PMID:local_780141fc, PMID:local_3a331973


#### Speech, voice and language *(n = 26)*

The group contains both the corpus's most privacy-considered design and clear instances of demographic confounding. Where a second clinical group is included, performance falls markedly.

*Representative studies:* PMID:local_76fc33d0, PMID:local_be8aa62f, PMID:local_7b3e6b3a, PMID:local_36aa3ace, PMID:local_5b1d0b36

*Caveats:* Modality assignment is approximate, derived from each record's charted data-type field rather than from a categorisation applied by the original authors.

**Group-specific Q&A:**

- **Q:** What does the group find when a clinical comparison group is added?
  **A:** local_1b457442 reports ROC-AUC 0.87-0.93 for ASD versus typical development and 0.71 once developmental delay is included, reading the gap as showing the risk of misdiagnosis.
  *Sources:* PMID:local_76fc33d0, PMID:local_be8aa62f, PMID:local_7b3e6b3a, PMID:local_36aa3ace, PMID:local_5b1d0b36
- **Q:** What confounds the binary results?
  **A:** Age. In the Nadig corpus used by local_1b457442 the ASD children average roughly two years older than the typically developing children, and age is one of the four features the paper identifies as carrying the result.
  *Sources:* PMID:local_76fc33d0, PMID:local_be8aa62f, PMID:local_7b3e6b3a, PMID:local_36aa3ace, PMID:local_5b1d0b36
- **Q:** What behavioural markers emerge?
  **A:** Mean length of utterance and mean length of turn ratio - the child's conversational share relative to the adult - both directionally interpretable and consistent with established clinical observation.
  *Sources:* PMID:local_76fc33d0, PMID:local_be8aa62f, PMID:local_7b3e6b3a, PMID:local_36aa3ace, PMID:local_5b1d0b36


#### Multimodal and other *(n = 19)*

Heterogeneous group containing the corpus's best-validated adult study and several of its clearest reporting failures.

*Representative studies:* PMID:local_7b468950, PMID:local_7547db68, PMID:local_6249bd63, PMID:local_8b9e0e8a, PMID:local_0e476686

*Caveats:* Modality assignment is approximate, derived from each record's charted data-type field rather than from a categorisation applied by the original authors.

**Group-specific Q&A:**

- **Q:** What does the best-validated study in this group report?
  **A:** local_7b19a959: 74.0% accuracy on an age- and gender-matched sample against 81.8% unmatched, with the matched figure reported as primary and the eight-point difference attributed explicitly to demographic features; 83% for the AQ alone and 92% combined; and a label-randomisation control that falls to chance.
  *Sources:* PMID:local_7b468950, PMID:local_7547db68, PMID:local_6249bd63, PMID:local_8b9e0e8a, PMID:local_0e476686
- **Q:** What are the group's principal failures?
  **A:** local_367db0b6 reports 25,000 positive cases drawn from a 10,000-sample dataset and describes the result as class-balanced, names an exemplary metric set and reports no value from it, and declares no ethics approval necessary in a paper premised on parental consent withdrawal for newborns' data. local_1d856900 splits an unsourced 100-sample clinical dataset after fourfold augmentation and reports 98.2% on 40 rows against 4,220 features.
  *Sources:* PMID:local_7b468950, PMID:local_7547db68, PMID:local_6249bd63, PMID:local_8b9e0e8a, PMID:local_0e476686


#### Facial images (still photographs) *(n = 16)*

Sixteen studies, all reusing the same web-scraped corpus of children's facial photographs, reporting 81.0% to 99.8%. Reported accuracy tracks model capacity and evaluation looseness rather than signal. No study performs a source-held-out evaluation, none establishes the number of distinct children, and none reports a diagnostic procedure for any image.

*Representative studies:* PMID:local_261afa51, PMID:local_948fa8d0, PMID:local_b9e05a8f, PMID:local_5a331ffe, PMID:local_0f5a351e

*Caveats:* Modality assignment is approximate, derived from each record's charted data-type field rather than from a categorisation applied by the original authors.

**Group-specific Q&A:**

- **Q:** What is the provenance of the data?
  **A:** A public corpus assembled from web searches and social-media pages with autism-related content, redistributed through Kaggle and Roboflow. Several papers state the provenance confound in their own methods sections and none tests it. Two describe the corpus as something it is not: local_986436f2 supplies age, ethnicity, urban/rural and facial-morphology annotations that the dataset does not contain and reports 2D photographs as 3D scans with 0.9 mm landmark error; local_44d3aef5 reports 547 images as 547 children.
  *Sources:* PMID:local_261afa51, PMID:local_948fa8d0, PMID:local_b9e05a8f, PMID:local_5a331ffe, PMID:local_0f5a351e
- **Q:** How are outcomes ascertained?
  **A:** By folder name. No study in this group reports a diagnostic instrument, criterion, assessor or date for any image.
  *Sources:* PMID:local_261afa51, PMID:local_948fa8d0, PMID:local_b9e05a8f, PMID:local_5a331ffe, PMID:local_0f5a351e
- **Q:** What does the group's own evidence say about what the models decide?
  **A:** local_e54c304c ran the only out-of-distribution probe in the branch: one of its two detectors classified a photograph of a dog as Autistic and a leaf as Nonautistic. Separately, local_e04bcd32's headline 96.94% is a resubstitution figure computed over all 2,940 images including the 2,540 used for training, and local_986436f2's confusion matrix sums to 300 cases against a stated 200-image evaluation.
  *Sources:* PMID:local_261afa51, PMID:local_948fa8d0, PMID:local_b9e05a8f, PMID:local_5a331ffe, PMID:local_0f5a351e
- **Q:** How is ethics handled?
  **A:** Mostly by exemption or silence. local_13c1f46d states that the study did not involve direct experiments on humans because publicly available Kaggle datasets were used; local_e4c2a776 declares that the article contains no studies with human participants. The exceptions are local_945300a5, which recruited and photographed 70 children under an ethics code with parental consent and teacher accompaniment, and local_e54c304c, which is the only study in the branch to state that given the child's facial data aspect, there is ethical sensitivity.
  *Sources:* PMID:local_261afa51, PMID:local_948fa8d0, PMID:local_b9e05a8f, PMID:local_5a331ffe, PMID:local_0f5a351e


#### Questionnaire and checklist data *(n = 15)*

Reported accuracy in this group is governed by how much of the label's own arithmetic survives preprocessing. Where all checklist items are retained the models reach 99-100%; where half are deleted performance falls to 0.83. No study in the group predicts a clinician diagnosis.

*Representative studies:* PMID:local_9c4af7a3, PMID:local_02fa4215, PMID:local_361b59fb, PMID:local_8ff20c48, PMID:local_ed30e24c

*Caveats:* Modality assignment is approximate, derived from each record's charted data-type field rather than from a categorisation applied by the original authors.

**Group-specific Q&A:**

- **Q:** What is the outcome variable?
  **A:** In every case a threshold on the sum of the items supplied as features. local_ed30e24c classified 292 of 292 correctly; local_183eaef4 reached accuracy 1.0000 with CARS, SRS, AQ-10 and Q-CHAT-10 scores all in the predictor set; local_8ff20c48 deleted five of ten items for correlating with the score they are summed into and fell to 0.83; local_14df790d showed that every subset containing the total or its summands performs well.
  *Sources:* PMID:local_9c4af7a3, PMID:local_02fa4215, PMID:local_361b59fb, PMID:local_8ff20c48, PMID:local_ed30e24c
- **Q:** Does any study identify the circularity?
  **A:** One. local_af9f4c12 states that because the severity labels are deterministically derived from the same behavioural attributes used as model inputs, the task constitutes a structured function-approximation setting and high performance should not be read as diagnostic accuracy - and repeats the restriction in five sections. local_14df790d comes closest without naming it, observing that retaining both the total and its items could mislead the model.
  *Sources:* PMID:local_9c4af7a3, PMID:local_02fa4215, PMID:local_361b59fb, PMID:local_8ff20c48, PMID:local_ed30e24c
- **Q:** What use is proposed for the outputs?
  **A:** local_b9de0d7f is the extreme case: a six-way teaching-method target generated with numpy from the Q-CHAT total, predicted at 0.9917 from the items that total is computed from, with an uncited severity-to-intervention mapping offered to educators as a basis for decisions about individual children's education.
  *Sources:* PMID:local_9c4af7a3, PMID:local_02fa4215, PMID:local_361b59fb, PMID:local_8ff20c48, PMID:local_ed30e24c


#### Robot-mediated and interactive-system data *(n = 8)*

Small group characterised by primary data collection in interactive settings, and containing the corpus's methodological benchmark for prospective validation.

*Representative studies:* PMID:local_cb9a5b48, PMID:local_9528024b, PMID:local_d587f9cd, PMID:local_c6f1209b, PMID:local_1579c85a

*Caveats:* Modality assignment is approximate, derived from each record's charted data-type field rather than from a categorisation applied by the original authors.

**Group-specific Q&A:**

- **Q:** What is the strongest design in the corpus?
  **A:** local_7b468950, the Play.Care phase III protocol (NCT03438994): a locked algorithm, blinded prospective external validation, sensitivity and specificity as prespecified primary outcomes, a 200-child cohort with other neurodevelopmental disorders for differential diagnosis, and a design powered to detect underperformance. It states of its own pilot that it only trained and tested patterns on that particular dataset.
  *Sources:* PMID:local_cb9a5b48, PMID:local_9528024b, PMID:local_d587f9cd, PMID:local_c6f1209b, PMID:local_1579c85a


#### Video of naturalistic behaviour *(n = 4)*

The most ecologically valid data in the corpus - spontaneous behaviour recorded in children's homes - undermined by clip-level partitioning and, in one case, by a task that contains no non-autistic comparison at all.

*Representative studies:* PMID:local_4b031d44, PMID:local_83b99e73, PMID:local_5b7ea7dc, PMID:local_3e88f3ce

*Caveats:* Modality assignment is approximate, derived from each record's charted data-type field rather than from a categorisation applied by the original authors.

**Group-specific Q&A:**

- **Q:** What is the validation problem?
  **A:** local_8d43288c formed folds over clips extracted from 75 parent-posted YouTube videos with no video- or child-level grouping, and read fold standard deviations below 0.004 as robustness.
  *Sources:* PMID:local_4b031d44, PMID:local_83b99e73, PMID:local_5b7ea7dc, PMID:local_3e88f3ce
- **Q:** What is actually being classified?
  **A:** In local_8d43288c, four movement categories within a dataset containing only autistic children - so the study's screening and diagnostic framing cannot follow from its result, as its own decision-support hedge implicitly concedes.
  *Sources:* PMID:local_4b031d44, PMID:local_83b99e73, PMID:local_5b7ea7dc, PMID:local_3e88f3ce


#### Physiological and thermal signals *(n = 2)*

Very small group; the charted members show the corpus's permissive end of the ethics gradient.

*Representative studies:* PMID:local_0b0960c4, PMID:local_45ef7022

*Caveats:* Modality assignment is approximate, derived from each record's charted data-type field rather than from a categorisation applied by the original authors.

**Group-specific Q&A:**

- **Q:** What ethics reporting appears?
  **A:** local_45ef7022 records Ethical Approval. Not required. for thermal imaging of 30 disabled children alongside proposed medication dosing.
  *Sources:* PMID:local_0b0960c4, PMID:local_45ef7022
