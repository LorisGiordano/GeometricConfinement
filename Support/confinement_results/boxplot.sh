cd $(dirname "$(realpath $0)")

source $(conda info --base)/etc/profile.d/conda.sh
conda activate statistics

DATASET="${1}"

if [[ "$DATASET" == "CTPel" ]]; then
    python Scripts/plots.py \
        --units px \
        --plot-radius-objective-trend \
        --output-dir Plots/CTPel \
        --model base3=Models/model_CTPel_det_3.json \
        --model seg3=Models/model_CTPel_seg_3.json \
        --model conf3=Models/model_CTPel_con_3.json \
        --model full3=Models/model_CTPel_full_3.json \
        --model base5=Models/model_CTPel_det_5.json \
        --model seg5=Models/model_CTPel_seg_5.json \
        --model conf5=Models/model_CTPel_con_5.json \
        --model full5=Models/model_CTPel_full_5.json \
        --model base7=Models/model_CTPel_det_7.json \
        --model seg7=Models/model_CTPel_seg_7.json \
        --model conf7=Models/model_CTPel_con_7.json \
        --model full7=Models/model_CTPel_full_7.json \
        --model base10=Models/model_CTPel_det_10.json \
        --model seg10=Models/model_CTPel_seg_10.json \
        --model conf10=Models/model_CTPel_con_10.json \
        --model full10=Models/model_CTPel_full_10.json \
        --model nnLandmark=Models/model_CTPel_nnLandmark.json
elif [[ "$DATASET" == "ImageTBAD" ]]; then
    python Scripts/plots.py \
        --units px \
        --plot-radius-objective-trend \
        --output-dir Plots/ImageTBAD \
        --model base3=Models/model_ImageTBAD_det_3.json \
        --model seg3=Models/model_ImageTBAD_seg_3.json \
        --model conf3=Models/model_ImageTBAD_con_3.json \
        --model full3=Models/model_ImageTBAD_full_3.json \
        --model base5=Models/model_ImageTBAD_det_5.json \
        --model seg5=Models/model_ImageTBAD_seg_5.json \
        --model conf5=Models/model_ImageTBAD_con_5.json \
        --model full5=Models/model_ImageTBAD_full_5.json \
        --model base7=Models/model_ImageTBAD_det_7.json \
        --model seg7=Models/model_ImageTBAD_seg_7.json \
        --model conf7=Models/model_ImageTBAD_con_7.json \
        --model full7=Models/model_ImageTBAD_full_7.json \
        --model base10=Models/model_ImageTBAD_det_10.json \
        --model seg10=Models/model_ImageTBAD_seg_10.json \
        --model conf10=Models/model_ImageTBAD_con_10.json \
        --model full10=Models/model_ImageTBAD_full_10.json \
        --model base20=Models/model_ImageTBAD_det_20.json \
        --model seg20=Models/model_ImageTBAD_seg_20.json \
        --model conf20=Models/model_ImageTBAD_con_20.json \
        --model full20=Models/model_ImageTBAD_full_20.json \
        --model nnLandmark=Models/model_ImageTBAD_nnLandmark.json
else
    echo "Invalid dataset name: $DATASET"
fi