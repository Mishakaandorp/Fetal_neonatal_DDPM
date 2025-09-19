#!/bin/bash

# # Define directories and file paths
# Change "Dataset" with your own dataset
# Remember to include both 'mri' and 'label'
Dataset='Example_fetal_labels'
L7label_BASE="/media/m-ssd4/Misha/Project_synthetic_brain/Datasets/Paper_code/Dataset/$Dataset/label"
T2_BASE="/media/m-ssd4/Misha/Project_synthetic_brain/Datasets/Paper_code/Dataset/$Dataset/mri" #optionally
SYNH_files_PER_SUB=2 # Defining number of synthetic files
# Which pathologies to synthetisize
Synthesize_pathology='noPNH' # 'ventriculomegaly' 'all_pathologies' 'noPNH' 'cortical_thickness'
save_files_name='synthesize_'$Synthesize_pathology'_'$SYNH_files_PER_SUB'_persub'
OUTPUTDIR="../manipulated_labels/$Dataset/$save_files_name/labels_per_subject/"
JSON_OUTPUTDIR="../manipulated_labels/$Dataset/$save_files_name/json_files/"
FINAL_IM_OUTPUTDIR="../manipulated_labels/$Dataset/$save_files_name/all_final_labels/"
FINAL_IM_OUTPUTDIR2="../generated_MRIs/$Dataset/$save_files_name/label/"

# f"../generated_MRIs/original_space/
mkdir -p "$OUTPUTDIR" "$JSON_OUTPUTDIR" "$FINAL_IM_OUTPUTDIR"


# Function to create JSON files for a patient
create_json_files() {
    local patient_id=$1
    local num_files=$2
    local Synthesize_pathology=$3
    
    echo "splitting hemis and ventricles for individual hemisphere ventriculomegaly processing..."
    python3 seperate_hemispheres_example.py \
        --subject "$patient_id" \
        --outputdir "$OUTPUTDIR"
    
    echo "Creating ${Synthesize_pathology} ${num_files} JSON files for patient ${patient_id}..."
    python3 generate_synthetic_pathology_json_files_example.py \
        --subject "$patient_id" \
        --synthetic_files_total "$num_files" \
        --outputdir "$OUTPUTDIR" \
        --outputdir_json_files "$JSON_OUTPUTDIR" \
        --synthesize_pathology "$Synthesize_pathology" 
}

copy_files() {
    echo coppying files

    local patient_id=$1
    local TARGET_DIR_7label=$2
    local TARGET_DIR_T2=$3
    # local TARGET_DIR_91label=$4
    
    local output_dir="${OUTPUTDIR}/${patient_id}/orig_files"
    mkdir -p $output_dir
    # Set the target directory and variable name from arguments

    file_T2=$(find "$TARGET_DIR_T2" -maxdepth 1 -type f -name "${patient_id}*")
    # Check if a file was found
    if [[ -n "$file_T2" ]]; then
        # Extract the whole filename
        filename=$(basename "$file_T2")
        echo "Found file: $filename"
        cp "${TARGET_DIR_T2}/${filename}" "${output_dir}/${patient_id}_T2w_orig.nii.gz"
    else
        echo "No file found starting with '$patient_id' in '$TARGET_DIR_T2'."
    fi

    # Find the first file that starts with the specified variable name
    file_7label=$(find "$TARGET_DIR_7label" -maxdepth 1 -type f -name "${patient_id}*")
    # Check if a file was found
    if [[ -n "$file_7label" ]]; then
        # Extract the whole filename
        filename=$(basename "$file_7label")
        echo "Found file: $filename"
        cp "${TARGET_DIR_7label}/${filename}" "${output_dir}/${patient_id}_7label_orig.nii.gz"
    else
        echo "No file found starting with '$patient_id' in '$TARGET_DIR_7label'."
    fi
}


# Function to align ventricles
generate_ventriculomegaly() {
    local patient_id=$1
    local vent_l=$2
    local vent_r=$3
    local input_label=$4
    local output_label=$5

    input_seg="${OUTPUTDIR}/${patient_id}/orig_files/${patient_id}_seg_orig.nii.gz"
    echo "Generating ventriculomegaly for ${patient_id}..."
    
    python3 manipulate_ventrikels_on_seperated_halfs_example.py \
        --outputdir "$OUTPUTDIR" \
        --subject "$patient_id" \
        --Ventriculomegaly_L "$vent_l" \
        --Ventriculomegaly_R "$vent_r" \
        --input_file_label "$input_label" \
        --output_label "$output_label"
}

# Function to generate PNH
generate_pnh() {
    local patient_id=$1
    local num_nodules=$2
    local input_file=$3
    local output_file=$4

    echo "Generating PNH with ${num_nodules} nodules for ${patient_id}..."
    python pnh_nodules_generation.py \
        --outputdir "$OUTPUTDIR" \
        --subject "$patient_id" \
        --number_nodules "$num_nodules" \
        --input_file "$input_file" \
        --output_file "$output_file"
    echo "PNH generated."
}

# Function to align cerebellum hypoplasia
generate_cerebellum_hypoplasia() {
    local patient_id=$1
    local shrink_factor=$2
    local input_file=$3
    local output_file=$4

    echo $input_file

    echo "Generating cerebellum hypoplasia for ${patient_id} with shrink factor ${shrink_factor}..."
    python cerebellum_shrinkage_example.py \
        --outputdir "$OUTPUTDIR" \
        --subject "$patient_id" \
        --shrink_factor_cerebellum "$shrink_factor" \
        --input_file "$input_file" \
        --output_file "$output_file"
    echo "Cerebellum hypoplasia generated."
}

# generate_cortical_thickness
generate_cortical_thickness() {
    local patient_id=$1
    local cortical_thickening=$2
    local input_file=$3
    local output_file=$4
    echo "cortical_thickening "$cortical_thickening" "
    
    echo "Generating thickening of the cortex for ${patient_id} with thickening: ${cortical_thickening}..."
    python cortical_thickening_example.py \
        --outputdir "$OUTPUTDIR" \
        --subject "$patient_id" \
        --cortical_thickening "$cortical_thickening" \
        --input_file "$input_file" \
        --output_file "$output_file"
    echo "Thickening cortex generated."
}

# Function to align pontocerebellar hypoplasias
generate_pontocerebellar_hypoplasias() {
    local patient_id=$1
    local scaling_factor=$2
    local input_transformed_file=$3
    local output_file=$4

    echo "Generating pontocerebellar hypoplasias for ${patient_id} with scaling factor ${scaling_factor}..."
    
    # Input and output file paths
    input_segmentation_dir="$OUTPUTDIR/$input_transformed_file"
    segmentation_scale="$OUTPUTDIR/$1/maps_scaling/"$1"_rec-mial_dseg_transformed_xz_scaling_"$2".nii.gz"
    output_segmentation_scale="$OUTPUTDIR/$1/maps_scaling/"$1"_rec-mial_dseg_transformed_xz_scaling_"$2".nii.gz"
    output_segmentation_shrink_brainstem="$OUTPUTDIR/$1/maps_scaling/'$1'_rec-mial_dseg_shrinked_brainstem'$2'.nii.gz"
    output_segmentation_shrink_brainstem_and_cerebellum="$1/maps_scaling/'$1'_rec-mial_dseg_shrinked_brainstem'$2'_and_cerebellum'$2'.nii.gz"

    local matrix_file="transformation_matrix1.mat"
    # Create transformation matrix for scaling brainstem
    cat <<EOF > "$matrix_file"
$scaling_factor 0.0 0.0 0.0
0.0 $scaling_factor 0.0 0.0
0.0 0.0 1.0 0.0
0.0 0.0 0.0 1.0
EOF

    mkdir -p $OUTPUTDIR/$1/maps_scaling/ &> /dev/null

    # Apply transformation
    echo "DOOOO flirt label image"
    flirt -in "$input_segmentation_dir" -ref "$input_segmentation_dir" -out "${output_segmentation_scale}" -applyxfm -init "$matrix_file" -paddingsize 0.0 -interp nearestneighbour
    echo "Applying transformations..."
    echo "Shrinking brainstem..."

    echo "${output_segmentation_shrink_brainstem}"

    python brainstem_shrinkage_example.py \
        --subject "$patient_id" \
        --outputdir "$OUTPUTDIR" \
        --shrink_factor "$scaling_factor" \
        --orig_file "$input_transformed_file" \
        --input_file "${segmentation_scale}" \
        --output_file "${output_segmentation_shrink_brainstem}"
    echo "Shrinking brainstem done"
    echo "Shrinking cerebellum..."
    python cerebellum_shrinkage_example.py \
        --outputdir "$OUTPUTDIR" \
        --subject "$patient_id" \
        --scaling_factor_brainstem "$scaling_factor" \
        --input_file "${output_segmentation_shrink_brainstem}" \
        --output_file "${output_segmentation_shrink_brainstem_and_cerebellum}"
    echo "Shrinking cerebellum done"
    # sleep 30
    echo "overwriting labels..."
    # To ensure the original labels are the same, except cerebellum, brainstem, ventricles, eCSF
    python overwrite_labels_except_eCSF_cerebellum_brainstem_example.py \
        --outputdir "$OUTPUTDIR" \
        --subject "$patient_id" \
        --orig_file "$input_transformed_file" \
        --input_file "${output_segmentation_shrink_brainstem_and_cerebellum}" \
        --output_file "$output_file"
    echo "overwriting labels done"

    echo "Pontocerebellar hypoplasias generated."
}

# Function to align microcephaly
generate_microcephaly() {
    local patient_id=$1
    local shrink_factor=$2
    local input_file=$3
    local output_file=$4

    echo "Generating microcephaly for ${patient_id} with shrink factor ${shrink_factor}..."
    python3 microcephaly_generation.py \
        --outputdir "$OUTPUTDIR" \
        --subject "$patient_id" \
        --shrink_factor "$shrink_factor" \
        --input_file "$input_file" \
        --output_file "$output_file"
    echo "Microcephaly generated."
}
#!/bin/bash

# Set the target directory to the first argument
# Loop through each file in the specified directory
for file in "$L7label_BASE"/*; do
    # Only process files (skip directories)
    if [[ -f "$file" ]]; then
        # Extract the base filename without the directory path
        filename=$(basename "$file")
        
        # Extract the part before the second underscore (_)
        dir_name=$(echo "$filename" | cut -d'_' -f1,2)
        base_name="${filename/_label.nii.gz/}"

        # Create a new directory with the extracted name if it doesn't exist
        if [[ ! -d "$OUTPUTDIR" ]]; then
            
            mkdir "$OUTPUTDIR"
            echo "Directory created: $OUTPUTDIR/"
        fi
        # Create a new directory with the extracted name if it doesn't exist
        if [[ ! -d "$OUTPUTDIR/$base_name" ]]; then

            mkdir "$OUTPUTDIR/$base_name"
            echo "Directory created: $OUTPUTDIR/$base_name"
        fi
    fi
done
echo created directories

# Loop through all directories in the specified folder

for dir in "$OUTPUTDIR"/*/; do
    # Only process if it's a directory
    if [[ -d "$dir" ]]; then
        # Get the directory name (strip the path)
        subject_id=$(basename "$dir")
        echo "Directory: $dir_name"
        echo "Processing ${subject_id}..."
        copy_files "$subject_id" "$L7label_BASE" "$T2_BASE" "$L91label_BASE"
        echo $Synthesize_pathology
        create_json_files "$subject_id" $SYNH_files_PER_SUB $Synthesize_pathology
    fi
done


for json_file in $JSON_OUTPUTDIR/*.json; do
    echo "Processing JSON file: $json_file"

    # Load Json file as variables
    subject=$(cat $json_file | sed -n 's/^[[:space:]]*"subject": "\([^"]*\)",/\1/p' | sed ':a;N;$!ba;s/\n//g' | sed 's/[^[:print:]]//g')
    vent_asym_L=$(cat $json_file | sed -n 's/^[[:space:]]*"ventriculomegaly_asymetric_L": \([^,]*\),/\1/p' | sed ':a;N;$!ba;s/\n//g' | sed 's/[^[:print:]]//g')
    vent_asym_R=$(cat $json_file | sed -n 's/^[[:space:]]*"ventriculomegaly_asymetric_R": \([^,]*\),/\1/p' | sed ':a;N;$!ba;s/\n//g' | sed 's/[^[:print:]]//g')
    hypo_cerebellum=$(cat $json_file | sed -n 's/^[[:space:]]*"hypo_cerebellum": \([^,]*\),/\1/p' | sed ':a;N;$!ba;s/\n//g' | sed 's/[^[:print:]]//g')
    cortical_thickness=$(cat $json_file | sed -n 's/^[[:space:]]*"cortical_thickness": \([^,]*\),/\1/p' | sed ':a;N;$!ba;s/\n//g' | sed 's/[^[:print:]]//g')
    hypo_pontocerebellar=$(cat $json_file | sed -n 's/^[[:space:]]*"hypo_pontocerebellar": \([^,]*\),/\1/p' | sed ':a;N;$!ba;s/\n//g' | sed 's/[^[:print:]]//g')
    microcephaly=$(cat $json_file | sed -n 's/^[[:space:]]*"microcephaly": \([^,]*\),/\1/p' | sed ':a;N;$!ba;s/\n//g' | sed 's/[^[:print:]]//g')
    pnh=$(cat $json_file | sed -n 's/^[[:space:]]*"pnh": \([^,]*\),/\1/p' | sed ':a;N;$!ba;s/\n//g' | sed 's/[^[:print:]]//g')
    synthetic_file=$(cat $json_file | sed -n 's/^[[:space:]]*"synthetic_file": \([^,]*\),/\1/p' | sed ':a;N;$!ba;s/\n//g' | sed 's/[^[:print:]]//g')

    # Print extracted information
    echo ""
    echo "JSON file info:"
    echo "File: $json_file"
    echo "Subject: $subject"
    echo "Ventriculomegaly Asymetric L: $vent_asym_L"
    echo "Ventriculomegaly Asymetric R: $vent_asym_R"
    echo "Hypo Cerebellum: $hypo_cerebellum"
    echo "Cortical Thickening: $cortical_thickness"
    echo "Hypo Pontocerebellar: $hypo_pontocerebellar"
    echo "pnh: $pnh"
    echo "Microcephaly: $microcephaly"
    echo "synthetic_file: $synthetic_file"
    echo ""


    input_file_label="${subject}/orig_files/${subject}_7label_orig.nii.gz"
    output_label_ventricles="${subject}/ventriculomegaly/${subject}_json_${synthetic_file}_rec-mial_dseg_Ventriculomegaly_L${vent_asym_L}_Ventriculomegaly_R_${vent_asym_R}.nii.gz"

    # IF statements identifies whether the label map manipulation should be performed. Else, copy the previous result..   
    # Generate ventricles
    if [ "$vent_asym_L" -eq 0 ] && [ "$vent_asym_R" -eq 0 ]; then
        echo No Ventriculomegaly
        echo coppying file..
        mkdir -p "$(dirname "$OUTPUTDIR/$output_label_ventricles")" &> /dev/null
        cp "$OUTPUTDIR/$input_file_label" "$OUTPUTDIR/$output_label_ventricles"
    else
        generate_ventriculomegaly "$subject" "$vent_asym_L" "$vent_asym_R" "$input_file_label" "$output_label_ventricles"
    fi
    echo ""

    pnh_output=''$subject'/pnh/'$subject'_json_'$synthetic_file'_rec-mial_dseg_pnh_'$pnh'.nii.gz'
    # Generate PNH if specified
    if [ "$pnh" -ne 0 ]; then
        generate_pnh "$subject" "$pnh" "$output_label_ventricles" "$pnh_output"
    else
        echo No PNH
        echo coppying file..
        
        mkdir -p "$(dirname "$OUTPUTDIR/$pnh_output")" &> /dev/null

        cp "$OUTPUTDIR/$output_label_ventricles" "$OUTPUTDIR/$pnh_output"
    fi
    cerebellum_output=''$subject'/cerebellum_hypoplasia/'$subject'_json_'$synthetic_file'_rec-mial_dseg_shrinked_cerebellum'$hypo_cerebellum'.nii.gz'
    echo ""

    # Generate cerebellum hypoplasia if specified
    if [ "$hypo_cerebellum" -eq 1  ]; then
        echo No cerebellum hypoplasia
        echo coppying file..

        mkdir -p "$(dirname "$OUTPUTDIR/$cerebellum_output")" &> /dev/null
        cp $OUTPUTDIR/"$pnh_output" $OUTPUTDIR/$cerebellum_output
    else
        cerebellum_input="$OUTPUTDIR/$pnh_output"
        generate_cerebellum_hypoplasia "$subject" "$hypo_cerebellum" "$cerebellum_input" "$cerebellum_output"
    fi


    coritcal_thickness_output=''$subject'/coritcal_thickness/'$subject'_json_'$synthetic_file'_rec-mial_dseg_shrinked_cerebellum'$hypo_cerebellum'.nii.gz'
    echo ""

    # Generate cortical thickening if specified
    echo "cortical_thickness"

    echo "$cortical_thickness"
    if [ "$cortical_thickness" -eq 0 ]; then
        echo No cortical thickening
        echo coppying file..

        mkdir -p "$(dirname "$OUTPUTDIR/$coritcal_thickness_output")" &> /dev/null
        cp $OUTPUTDIR/$cerebellum_output $OUTPUTDIR/$coritcal_thickness_output
    else
        echo "$cortical_thickness"

        generate_cortical_thickness "$subject" "$cortical_thickness" "$cerebellum_output" "$coritcal_thickness_output"

    fi

    pontocerebellar_output=''$subject'/output_label_hypo_pontocerebellar/'$subject'_json_'$synthetic_file'_rec-mial_dseg_hypo_pontocerebellar_'$hypo_pontocerebellar'.nii.gz'
    echo ""
    # Generate pontocerebellar hypoplasias if specified
    if [ "$hypo_pontocerebellar" -eq 1  ]; then
        echo No pontocerebellar hypoplasia
        echo coppying file..
        mkdir -p "$(dirname "$OUTPUTDIR/$pontocerebellar_output")" &> /dev/null
        cp $OUTPUTDIR/$coritcal_thickness_output "$OUTPUTDIR/$pontocerebellar_output"
    else
        generate_pontocerebellar_hypoplasias "$subject" "$hypo_pontocerebellar" "$cerebellum_output" "$pontocerebellar_output"
    fi
    echo ""
    microcephaly_output=''$subject'/microephaly_maps/'$subject'_json_'$synthetic_file'_rec-mial_dseg_shrinked_brain_'$microcephaly'.nii.gz'

    # Generate microcephaly if specified
    if [ "$microcephaly" -eq 1 ]; then
        echo No microcephaly hypoplasia
        echo coppying file..
        mkdir -p "$(dirname "$OUTPUTDIR/$microcephaly_output")" &> /dev/null
        cp "$OUTPUTDIR/$pontocerebellar_output" "$OUTPUTDIR/$microcephaly_output"

    else
        generate_microcephaly "$subject" "$microcephaly" "$pontocerebellar_output" "$microcephaly_output"
    fi

    echo ""
    final_map=''$subject'/final_maps/'$subject'_json_'$synthetic_file'_L_'$vent_asym_L'_R_'$vent_asym_R'_cer_'$hypo_cerebellum'_pont_'$hypo_pontocerebellar'_mic_'$microcephaly'_pnh_'$pnh'_cor_'$cortical_thickness'.nii.gz'
    mkdir -p "$(dirname "$OUTPUTDIR/$final_map")"

    
    echo creating final map 1..

    cp $OUTPUTDIR/$microcephaly_output $OUTPUTDIR/$final_map
    
    mkdir -p $OUTPUTDIR/ &> /dev/null

    echo creating final map 2..
    cp $OUTPUTDIR/$final_map $FINAL_IM_OUTPUTDIR/
    cp $OUTPUTDIR/$final_map $FINAL_IM_OUTPUTDIR2/

    echo ""

    echo "Subject: $subject File: $json_file Finished"
    echo ""

done
echo ""

echo "All label manipulations completed."

echo ""

echo "Preprocess manipulated labels for image synthesis .."

python preprocess_labels_for_image_synthesis.py \
    --base_dataset "$Dataset" \
    --synthesize_option "$save_files_name"

echo "Labels manipulated for image synthesis "

mkdir -p "../../generated_MRIs/160_space/$Dataset/$save_files_name/"

cd ../Fetal_Neonatal_DDPM/scripts/
echo "Current working directory: $(pwd)"

python3 ../sample.py --inputfolder "../../preprocessing_manipulated_labels/$Dataset/$save_files_name/label_160_space/" --exportfolder "../../generated_MRIs/160_space/$Dataset/$save_files_name/" --fix_seed --input_size 160 --depth_size 160 --num_channels 64 --num_res_blocks 1 --batchsize 1 --num_samples 1 --num_class_labels 4 --timesteps 1000 --weightfile "../Trained_models/Fetal_Neonatal_DDPM_500kepochs.pt"  

cd ../../scripts/

echo "upscaling MRIs to original space .."

# Note - this only works of the MRI is available..
python resize_rescale_synthetic_image_to_original_shape.py \
    --base_dataset "$Dataset" \
    --synthesize_option "$save_files_name" \
    --T2_images "$T2_BASE" \
    --SYNH_files_PER_SUB "$SYNH_files_PER_SUB"

echo "All processing done."

