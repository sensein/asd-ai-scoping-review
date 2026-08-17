"""Shared categorization codebook used by Results and ICR analyses.

Category labels and patterns here are extracted from the finalized Results logic.
Keep changes explicit because they can alter manuscript counts and reliability values.
"""

from __future__ import annotations


TASK_TYPE_PATTERNS = {
    "gaze_visual_attention_task": (
        r"\bjoint[- ]?attention\b|\bjointattention\b|\bgaze\b|\beye[- ]?tracking\b|\bsaccade\w*\b|"
        r"\bscan[- ]?path\w*\b|\bscanpath\w*\b|\bwatch\w*\b|\bvideo\w*\b|\bmovie\w*\b|"
        r"\bmovie clip\w*\b|\bpicture\w*\b|\bscene\w*\b|\blook\w*\b|\bview\w*\b|\bobserve\w*\b|"
        r"\bbrows\w*\b|\bweb[- ]?search\w*\b|\bwebsite\w*\b|\bface viewing\b|\bvisual attention\b|"
        r"\bvisual exploration\b|\bobserving images\b|\bimage classification\b"
    ),
    "motor_movement_task": (
        r"\bplay\w*\b|\btoy\w*\b|\bwalk\w*\b|\bgait\b|\bmove\w*\b|\bmovement\w*\b|\bstand\w*\b|"
        r"\bstood\b|\breach\w*\b|\bgrasp\w*\b|\bpose\w*\b|\bgesture\w*\b|\bimitation\b|"
        r"\bimitat\w*\b|\bmotor\b|\bdrag\b"
    ),
    "language_speech_audio_task": (
        r"\brepl\w*\b|\bspeak\w*\b|\bspoke\b|\blisten\w*\b|\bdiscuss\w*\b|\bconversation\w*\b|"
        r"\baudio\w*\b|\bsound\w*\b|\bspeech\b|\bdialog\w*\b|\bread\w*\b|\binterview\w*\b|"
        r"\bvocal\w*\b|\bvoice\b"
    ),
    "questionnaire_survey_task": (
        r"\bquestionnaire\w*\b|\bquestionnare\w*\b|\bquesstionnaire\w*\b|\bquesstionaire\w*\b|"
        r"\bques\w*\b|\bsurvey\w*\b|\bself[- ]?report\b|\bparent[- ]?report\b|\bcaregiver[- ]?report\b|"
        r"\brating scale\w*\b|\bq-chat-10\b|\bq-chat\b|\behr\b"
    ),
    "facial_emotion_expression_task": (
        r"\bfacial emotion\w*\b|\bfacial expression\w*\b|\bemotion recognition\b|\bemotion identification\b|"
        r"\bidentify\w* emotion\w*\b|\bidentifying emotion\w*\b|\brecogniz\w* emotion\w*\b|"
        r"\bemotions? from facial image\w*\b|\bfacial image\w*\b|\bface recognition\b|\bfaze recognition\b|"
        r"\bfacial affect\b|\bimages of participants\b|\bface\b"
    ),
    "social_interaction_task": (
        r"\binteract\w*\b|\bsocial\b|\brelation\w*\b|\brobot\w*\b|\bvirtual reality\b|\bvirtualreality\b|"
        r"\bvr\b|\bsocial interaction\b|\bjoint activity\b|\btweets\b|\bcommunication\b|\bresponse to name\b"
    ),
    "decision_making_cognitive_task": (
        r"\bdecision[- ]?making\b|\bchoice task\b|\bcognitive task\b|\brisk task\b|\breward task\b|\breaction time\b"
    ),
    "clinical_observation_assessment_task": (
        r"\bados\b|\bados[- ]?2\b|\bclinical observation\b|\bdiagnostic observation\b|\bassessment task\b|"
        r"\bstructured assessment\b"
    ),
    "neurophysiology_neuroimaging_task": (
        r"\beeg\b|\berp\b|\bmri\b|\bfmri\b|\bpet\b|\bemg\b|\becg\b|\bbrain activity\b|\bbrain imaging\b|"
        r"\bneuroimaging\b|\bphysiolog\w*\b|\bbiosignal\w*\b|\bresting[- ]?state\b|\btask[- ]?state\b|"
        r"\belectrode\w*\b"
    ),
}

TASK_TYPE_COLS = list(TASK_TYPE_PATTERNS)

NOT_GIVEN_TASK_PATTERN = (
    r"^\s*$|^\s*-$|^\s*--$|^\s*not given\s*$|^\s*not reported\s*$|^\s*not specified\s*$|"
    r"^\s*none\s*$|^\s*no\s*$|^\s*na\s*$|^\s*n/a\s*$|^\s*n\.a\s*$|^\s*nd\s*$|"
    r"^\s*n/d\s*$|^\s*n\.d\s*$|^\s*nan\s*$"
)

LEARNING_TYPE_PATTERNS = {
    "supervised_learning": r"\bsupervised\b|\bsl\b|\bclassification\b|\bclassifier\b|\bregression\b",
    "unsupervised_learning": r"\bunsupervised\b|\bclustering\b|\bk[- ]?means\b|\bpca\b|\bautoencoder\b|\bvae\b",
    "reinforcement_learning": r"\breinforcement\b|\brl\b|\bq[- ]?learning\b|\bpomdp\b|\bmarkov decision\b",
    "semi_self_or_transfer_learning": r"\bsemi[- ]?supervised\b|\bself[- ]?supervised\b|\btransfer learning\b|\bfine[- ]?tuning\b|\bpre[- ]?trained\b",
}

ALGORITHM_FAMILY_PATTERNS = {
    "classical_machine_learning_models": (
        r"linear regres+sion|logistic regression|linear discriminant analysis|\blda\b|quadratic classifier|"
        r"support vector machine|\bsvm\b|\bknn\b|k[- ]?nearest neighbors?|naive[- ]?bayes|naïve[- ]?bayes|\bnb\b|"
        r"decision tree|random forest|extra trees|regulari[sz]ed greedy forest|\bcart\b|\bridge\b|elastic net"
    ),
    "ensemble_models": (
        r"gradient boost|gradient boosting|\bgb\b|\bgbm\b|\bgbdt\b|adaboost|ada boost|xgboost|"
        r"extreme gradient boosting|lightgbm|light gbm|\blgbm\b|catboost|cat boost|ensemble\w*|voting|bagging|"
        r"boosting|stacking|stacked ensemble"
    ),
    "neural_network_models": (
        r"\bann\b|artificial neural network|multi[- ]?layer perceptron|multilayer perceptron|\bmlp\b|\bfnn\b|"
        r"feed[- ]?forward|fcdnn|\bdnn\b|deep neural network|\bcnn\b|convolutional neural network|resnet|"
        r"resnet[- ]?50|googlenet|inception|inceptionv3|vgg|vgg[- ]?16|vgg[- ]?19|mobilenet|efficientnet|"
        r"xception|convnext|yolo|yolov8|neural network|\brnn\b|recurrent neural network|\blstm\b|bi[- ]?lstm|"
        r"\bblstm\b|\bgru\b|cnn[-+ ]?gru|cnn[-+ ]?lstm|attention|relu|dropout|fully connected|softmax|"
        r"graph convolutional network|\bgcn\b|graph neural network|\bgnn\b|msg3d|st[- ]?gcn|ksnet|"
        r"generative adversarial network|\bgan\b|\bvae\b|sdae|stacked denoising autoencoder|autoencoder|"
        r"binary classifier|pnn|transformer|bert|wav2vec"
    ),
    "statistical_and_other_specialised_models": (
        r"\blasso\b|kernel extreme learning machine|kernel extreme machine learning|\bkelm\b|"
        r"extreme learning machine|\belm\b|fvelm|markov model|\bpomdp\b|\bhmm\b|hidden markov|bayesian|"
        r"gaussian process|gami[- ]?net|giza pyramids construction|\bgpc\b|metaheuristic|genetic algorithm|"
        r"particle swarm|\bpso\b"
    ),
}

HYBRID_MODEL_PATTERN = (
    r"\b(?:cnn|lstm|dnn|ann|rnn|gru|blstm|bi[- ]?lstm|mlp|svm|pnn|autoencoder|ae|vgg|vgg[- ]?16|"
    r"vgg[- ]?19|resnet|googlenet|inception|gcn|gnn|gan|vae|bert|transformer|xgboost|random forest|"
    r"decision tree|dt|knn|naive bayes|nb|kelm|elm)\b\s*(?:\+|&|and|with|/|-)\s*"
    r"\b(?:cnn|lstm|dnn|ann|rnn|gru|blstm|bi[- ]?lstm|mlp|svm|pnn|autoencoder|ae|vgg|vgg[- ]?16|"
    r"vgg[- ]?19|resnet|googlenet|inception|gcn|gnn|gan|vae|bert|transformer|xgboost|random forest|"
    r"decision tree|dt|knn|naive bayes|nb|kelm|elm)\b|hybrid model|hybrid framework|hybrid architecture|"
    r"hybrid approach|\bhybrid\b|dual[- ]?stream|multi[- ]?stream|two[- ]?stream|ensemble of|"
    r"combination of models|combined model|combined models"
)

EVALUATION_METRIC_PATTERNS = {
    "accuracy": r"\baccuracy\b|\bbalanced accuracy\b|\bclassification accuracy\b",
    "specificity": r"\bspecificity\b|\btnr\b|\btrue negative rate\b",
    "sensitivity_recall": r"\bsensitivity\b|\btpr\b|\btrue positive rate\b|\brecall\b",
    "precision_ppv": r"\bprecision\b|\bpositive predictive value\b|\bppv\b",
    "f1_score": r"f[- ]?1|f1 score|f[- ]?measure|f measure",
    "auc_roc": r"\bauc\b|\broc\b|\bauc[- ]?roc\b|\bau[- ]?roc\b|\bauroc\b|\barea under the curve\b|\breceiver operating characteristic\b",
    "other_evaluation_reporting_metrics": (
        r"\bconfusion matrix\b|\bconfusion\b|\bclassification report\b|\berror matrix\b|\berror[- ]?rate\b|"
        r"\bclassification error\b|\bmae\b|\bmse\b|\brmse\b|\bloss\b|\bcross[- ]?entropy\b|"
        r"\bmatthews correlation coefficient\b|\bmcc\b|\bnegative predictive value\b|\bnpv\b|\buar\b|"
        r"\bkappa\b|\bg[- ]?mean\b|\bbalanced error\b|\bdiagnostic validity\b"
    ),
}

OTHER_ACCURACY_METRIC_PATTERN = (
    r"\b(?:sensitivity|specificity|recall|precision|auc|auroc|roc|f[- ]?1|matthews correlation coefficient|mcc|"
    r"error[- ]?rate|positive predictive value|ppv|negative predictive value|npv|uar|tpr|tnr|kappa|"
    r"diagnostic validity|f[- ]?measure|g[- ]?mean|loss|mae|mse|rmse|pearson|correlation)\b"
)
