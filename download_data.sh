#!/bin/bash  

FOLDER_DATA_RAW=datat_raw
FOLDER_DATA_RAW_LABELS="${FOLDER_DATA_RAW}/labels/"
FOLDER_DATA_RAW_DATA="${FOLDER_DATA_RAW}/images/"

FILENAME_LABELS=ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip
FILENAME_DATA=ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip

FILENAME_LABELS_DST=ground_truth.zip
FILENAME_DATA_DST=image_data.zip

USERNAME=$1
PASSWORD=$2

echo "Username: $USERNAME";
echo "Password: $PASSWORD";


echo "Downloading ${FILENAME_LABELS} ..."
URL=ftp://$USERNAME:$PASSWORD@ftp.ipi.uni-hannover.de/ISPRS_BENCHMARK_DATASETS/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip
echo "url: $URL";
wget -O $FILENAME_LABELS_DST $URL
mkdir -p $FOLDER_DATA_RAW_LABELS
unzip -qq -o $FILENAME_LABELS_DST -d $FOLDER_DATA_RAW_LABELS
rm $FILENAME_LABELS_DST

echo "Downloading ${FILENAME_DATA} ..."
URL=ftp://$USERNAME:$PASSWORD@ftp.ipi.uni-hannover.de/ISPRS_BENCHMARK_DATASETS/Vaihingen/ISPRS_semantic_labeling_Vaihingen.zip
echo "url: $URL";
wget -O $FILENAME_DATA_DST $URL 
mkdir -p $FOLDER_DATA_RAW_DATA
unzip -qq -o $FILENAME_DATA_DST -d $FOLDER_DATA_RAW_DATA
rm $FILENAME_DATA_DST