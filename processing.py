import os
import polars as pl
import json
import random
from polars import DataFrame


def label_sentences(sentences: list[str], timestamps: list[dict]):
    speaking_durations = []  # speaking_durations[i] corresponds to sentences[i]

    tsIndx = 0
    for s in sentences:
        begin_time = timestamps[tsIndx]['begin_time']

        while tsIndx < len(timestamps) and timestamps[tsIndx]['phrase'][-1] != ".":
            tsIndx += 1

        if timestamps[tsIndx]['phrase'][-1] == ".":
            end_time = timestamps[tsIndx]['end_time']
            speaking_durations.append(
                {'sentence': s, 'begin_time': begin_time, 'end_time': end_time,
                    'duration': round(end_time - begin_time, 1)}
            )

        tsIndx += 1

    return speaking_durations


def get_transcription_abnormality_sentences_with_timestamps(file_path: str):
    transcript_path = os.path.join(file_path, "transcript.json")
    transcript = json.load(open(transcript_path))

    timestamps = transcript['time_stamped_text']
    # remove leading and trailing white space
    full_text = transcript['full_text'].strip().lower()

    sentences = full_text.split(". ")  # split into individual sentences

    for i, s in enumerate(sentences):
        if s[-1] != ".":
            sentences[i] = s + "."

    sentence_with_timestamps = label_sentences(sentences, timestamps)

    abnormality_sentences_with_timestamps = []

    for swt in sentence_with_timestamps:
        # remove trailing period
        sentence = swt['sentence'][:len(swt['sentence']) - 1]
        words = set(sentence.split(" "))
        if "no" not in words and "not" not in words:
            abnormality_sentences_with_timestamps.append(swt)

    return abnormality_sentences_with_timestamps


def create_class1_perceptual_error(csv_path: str, removed_sentence_indx: int, abnormality_sentences_with_timestamps: list[dict]):
    """
    Missed abnormality due to removed transcription sentence
    """
    initial_fixations_df = pl.read_csv(os.path.join(csv_path, "fixations.csv"))

    print(abnormality_sentences_with_timestamps[
        removed_sentence_indx])
    print("Initial fixations:", initial_fixations_df.shape)

    removed_sentence_begin_time = abnormality_sentences_with_timestamps[
        removed_sentence_indx]['begin_time']
    removed_sentence_end_time = abnormality_sentences_with_timestamps[
        removed_sentence_indx]['end_time']

    missed_fixations_df = initial_fixations_df.filter((pl.col('Time (in secs)') >= removed_sentence_begin_time) & (
        pl.col('Time (in secs)') <= removed_sentence_end_time))
    print("Missed fixations:", missed_fixations_df.shape)

    remaining_fixations_df = initial_fixations_df.filter(~((pl.col('Time (in secs)') >= removed_sentence_begin_time) & (
        pl.col('Time (in secs)') <= removed_sentence_end_time)))
    print("Remaining fixations:", remaining_fixations_df.shape)

    return missed_fixations_df, remaining_fixations_df


def half_eye_gaze_fixation(FPOGD):
    return FPOGD * 0.5


def create_class2_perceptual_error(csv_path: str, removed_sentence_indx: int, abnormality_sentences_with_timestamps: list[dict], fixation_reducer):
    """
    Missed abnormality due to reduced fixation duration
    """
    initial_fixations_df = pl.read_csv(os.path.join(csv_path, "fixations.csv"))

    print(abnormality_sentences_with_timestamps[
        removed_sentence_indx])

    removed_sentence_begin_time = abnormality_sentences_with_timestamps[
        removed_sentence_indx]['begin_time']
    removed_sentence_end_time = abnormality_sentences_with_timestamps[
        removed_sentence_indx]['end_time']

    fixations_reduced_df = initial_fixations_df.filter(
        (pl.col('Time (in secs)') >= removed_sentence_begin_time) &
        (pl.col('Time (in secs)') <= removed_sentence_end_time)
    )

    print(fixations_reduced_df['Time (in secs)', 'FPOGD'])

    reduced_fixations_output_df = initial_fixations_df.with_columns(
        pl.when((pl.col('Time (in secs)') >= removed_sentence_begin_time)
                & (pl.col('Time (in secs)') <= removed_sentence_end_time))
        .then(fixation_reducer(pl.col('FPOGD')))
        .otherwise(pl.col('FPOGD'))
        .alias('FPOGD')
    )

    print(reduced_fixations_output_df.filter(
        (pl.col('Time (in secs)') >= removed_sentence_begin_time) &
        (pl.col('Time (in secs)') <= removed_sentence_end_time)
    )['Time (in secs)', 'FPOGD'])

    return fixations_reduced_df, reduced_fixations_output_df


def create_both_class1_and_class2_perceptual_error(csv_path: str, abnormality_sentences_with_timestamps: list[dict], fixation_reducer):
    """
    Missed abnormality due to removed transcription sentence and reduced fixation duration
    """
    if len(abnormality_sentences_with_timestamps) == 1:
        return {}

    c1_removed_sentence_indx = random.randint(
        0, len(abnormality_sentences_with_timestamps) - 1)
    c2_removed_sentence_indx = c1_removed_sentence_indx

    while c2_removed_sentence_indx == c1_removed_sentence_indx:
        c2_removed_sentence_indx = random.randint(
            0, len(abnormality_sentences_with_timestamps) - 1)

    initial_fixations_df = pl.read_csv(os.path.join(csv_path, "fixations.csv"))
    print("Initial Fixations:", initial_fixations_df.shape)

    #! Add class 1 perceptual error
    c1_removed_sentence_begin_time = abnormality_sentences_with_timestamps[
        c1_removed_sentence_indx]['begin_time']
    c1_removed_sentence_end_time = abnormality_sentences_with_timestamps[
        c1_removed_sentence_indx]['end_time']

    print("Removed Class 1 sentence:", abnormality_sentences_with_timestamps[
        c1_removed_sentence_indx])

    missed_fixations_df = initial_fixations_df.filter((pl.col('Time (in secs)') >= c1_removed_sentence_begin_time) & (
        pl.col('Time (in secs)') <= c1_removed_sentence_end_time))

    print("Missed fixations:", missed_fixations_df.shape)

    remaining_fixations_df = initial_fixations_df.filter(~((pl.col('Time (in secs)') >= c1_removed_sentence_begin_time) & (
        pl.col('Time (in secs)') <= c1_removed_sentence_end_time)))

    print("Removed fixations:", remaining_fixations_df.shape)

    #! Add class 2 perceptual error
    c2_removed_sentence_begin_time = abnormality_sentences_with_timestamps[
        c2_removed_sentence_indx]['begin_time']
    c2_removed_sentence_end_time = abnormality_sentences_with_timestamps[
        c2_removed_sentence_indx]['end_time']

    print("Removed Class 2 sentence:", abnormality_sentences_with_timestamps[
        c2_removed_sentence_indx])

    fixations_reduced_df = remaining_fixations_df.filter(
        (pl.col('Time (in secs)') >= c2_removed_sentence_begin_time) &
        (pl.col('Time (in secs)') <= c2_removed_sentence_end_time)
    )

    print("Fixations reduced", fixations_reduced_df['Time (in secs)', 'FPOGD'])

    final_fixations_output_df = remaining_fixations_df.with_columns(
        pl.when((pl.col('Time (in secs)') >= c2_removed_sentence_begin_time)
                & (pl.col('Time (in secs)') <= c2_removed_sentence_end_time))
        .then(fixation_reducer(pl.col('FPOGD')))
        .otherwise(pl.col('FPOGD'))
        .alias('FPOGD')
    )

    print(final_fixations_output_df.filter(
        (pl.col('Time (in secs)') >= c2_removed_sentence_begin_time) &
        (pl.col('Time (in secs)') <= c2_removed_sentence_end_time)
    )['Time (in secs)', 'FPOGD'])

    print("Final output:", final_fixations_output_df.shape)
    # print("=====================================================")

    return c1_removed_sentence_indx, c2_removed_sentence_indx, missed_fixations_df, remaining_fixations_df, fixations_reduced_df, final_fixations_output_df


def create_class3_perceptual_error(csv_path: str):
    """
    Missed abnormality due to less experience
    """
    return pl.read_csv(os.path.join(csv_path, "fixations.csv"))


def get_correct_data(csv_path: str):
    return pl.read_csv(os.path.join(csv_path, "fixations.csv"))


def convert_df_to_dict(df: DataFrame):
    keys_to_extract = set({'FPOGX', 'FPOGY', 'FPOGD', 'Time (in secs)'})
    df_dict = df.to_dict(as_series=False)
    return {key: df_dict[key] for key in keys_to_extract if key in df_dict}


def main():
    """
    Divide number of samples into 
    """
    perceptual_error_class_labels = {
        1: 'Missed abnormality due to removed transcription sentence',
        2: 'Missed abnormality due to reduced fixation duration',
        3: 'Missed abnormality due to less experience'
    }

    fixation_fpath = "fixations"
    audio_seg_transcript_fpath = "audio_segmentation_transcripts"
    dicom_ids = os.listdir(fixation_fpath)

    # key: dicom_id: {correct_data, incorrect_data_class_label...}
    aggregated_dicom_data = {}
    aggregated_dicom_perceptual_error_info = {}  # key: dicom_id: {error_info...}
    # key: dicom_id: transcript_timestamps
    dicom_abnormality_transcript_timestamps = {}

    """
    50 samples divided into 10 samples 
    1. Missed abnormality due to removed transcription sentence
    2. Missed abnormality due to reduced fixation duration
    3. Missed abnormality due to less experience
    4. Missed abnormality due to removed transcription sentence and reduced fixation duration
    5. No error
    """
    num_samples = 50
    subgroup_ratios = {
        'no_error_ratio': 0.2,
        'class1_ratio': 0.2,
        'class2_ratio': 0.2,
        'class1_and_2_ratio': 0.2,
        'class3_ratio': 0.2,
    }

    subgroup_samples = []
    subgroup_labels = ['no_error', 'class1',
                       'class2', 'class1_and_2', 'class3']
    for k, ratio in subgroup_ratios.items():
        subgroup_samples.append(int(num_samples * ratio))

    # subgroup_samples = [1]
    # subgroup_labels = ['class1_and_2']

    curr_subgroup_indx = 0
    dicom_id_indx = 0

    for sg in subgroup_samples:
        if dicom_id_indx >= len(dicom_ids):
            break

        for _ in range(sg):
            if dicom_id_indx >= len(dicom_ids):
                break

            current_dicom_id = dicom_ids[dicom_id_indx]
            transcript_path = os.path.join(
                audio_seg_transcript_fpath, current_dicom_id)
            fixation_csv_path = os.path.join(
                fixation_fpath, current_dicom_id)

            curr_dicom_id_data = {
                'correct_data': {},
                'class_label_1_data': {},
                'class_label_2_data': {},
                'class_label_1_and_2_data': {},
                'class_label_3_data': {},
            }

            curr_dicom_id_perceptual_error_details = {
                'class_label_1': 0,
                'class_label_2': 0,
                'class_label_3': 0,
                # [{'class_label_1': 0, 'class_label_2': 0, 'class_label_3': 0, 'phrase': "", 'begin_time': 0, 'end_time': 0}]
                'phrases_removed': [],
                'missed_fixation_points': {},
                'fixation_points_reduced': {},
            }

            abnormality_sentences_with_timestamps = get_transcription_abnormality_sentences_with_timestamps(
                transcript_path)
            dicom_abnormality_transcript_timestamps[current_dicom_id
                                                    ] = abnormality_sentences_with_timestamps
            print(abnormality_sentences_with_timestamps)

            removed_sentence_indx = random.randint(
                0, len(abnormality_sentences_with_timestamps) - 1)

            print(
                "================================================================================")
            print("Dicom Id:", current_dicom_id)
            print("Perceptual Error Class Label:",
                  subgroup_labels[curr_subgroup_indx])

            correct_data_df = get_correct_data(fixation_csv_path)
            curr_dicom_id_data['correct_data'] = convert_df_to_dict(
                correct_data_df)

            if subgroup_labels[curr_subgroup_indx] == 'class1':

                c1_missed_fixations_df, c1_remaining_fixations_output_df = create_class1_perceptual_error(
                    fixation_csv_path, removed_sentence_indx, abnormality_sentences_with_timestamps)
                c1_abnormality_removed_sentence_info = {
                    'class_label_1': 1,
                    'class_label_2': 0,
                    'class_label_3': 0,
                    'phrase': abnormality_sentences_with_timestamps[removed_sentence_indx]['sentence'],
                    'begin_time': abnormality_sentences_with_timestamps[removed_sentence_indx]['begin_time'],
                    'end_time': abnormality_sentences_with_timestamps[removed_sentence_indx]['end_time'],
                }

                curr_dicom_id_data['class_label_1_data'] = convert_df_to_dict(
                    c1_remaining_fixations_output_df)
                curr_dicom_id_perceptual_error_details['class_label_1'] = 1
                curr_dicom_id_perceptual_error_details['phrases_removed'].append(
                    c1_abnormality_removed_sentence_info)
                curr_dicom_id_perceptual_error_details['missed_fixation_points'] = convert_df_to_dict(
                    c1_missed_fixations_df)
            elif subgroup_labels[curr_subgroup_indx] == 'class2':

                c2_reduced_fixations_df, c2_reduced_fixations_output_df = create_class2_perceptual_error(fixation_csv_path, removed_sentence_indx,
                                                                                                         abnormality_sentences_with_timestamps, half_eye_gaze_fixation)
                c2_abnormality_removed_sentence_info = {
                    'class_label_1': 0,
                    'class_label_2': 1,
                    'class_label_3': 0,
                    'phrase': abnormality_sentences_with_timestamps[removed_sentence_indx]['sentence'],
                    'begin_time': abnormality_sentences_with_timestamps[removed_sentence_indx]['begin_time'],
                    'end_time': abnormality_sentences_with_timestamps[removed_sentence_indx]['end_time'],
                }

                curr_dicom_id_data['class_label_2_data'] = convert_df_to_dict(
                    c2_reduced_fixations_output_df)
                curr_dicom_id_perceptual_error_details['class_label_2'] = 1
                curr_dicom_id_perceptual_error_details['phrases_removed'].append(
                    c2_abnormality_removed_sentence_info)
                curr_dicom_id_perceptual_error_details['fixation_points_reduced'] = convert_df_to_dict(
                    c2_reduced_fixations_df)
            elif subgroup_labels[curr_subgroup_indx] == 'class1_and_2':
                c1_removed_sentence_indx, c2_removed_sentence_indx, c1_c2_missed_fixations_df, c1_c2_remaining_fixations_df, c1_c2_fixations_reduced_df, c1_c2_final_fixations_output_df = create_both_class1_and_class2_perceptual_error(
                    fixation_csv_path, abnormality_sentences_with_timestamps, half_eye_gaze_fixation)
                curr_dicom_id_data['class_label_1_and_2_data'] = convert_df_to_dict(
                    c1_c2_final_fixations_output_df)

                c1_abnormality_removed_sentence_info = {
                    'class_label_1': 1,
                    'class_label_2': 0,
                    'class_label_3': 0,
                    'phrase': abnormality_sentences_with_timestamps[c1_removed_sentence_indx]['sentence'],
                    'begin_time': abnormality_sentences_with_timestamps[c1_removed_sentence_indx]['begin_time'],
                    'end_time': abnormality_sentences_with_timestamps[c1_removed_sentence_indx]['end_time'],
                }
                c2_abnormality_removed_sentence_info = {
                    'class_label_1': 0,
                    'class_label_2': 1,
                    'class_label_3': 0,
                    'phrase': abnormality_sentences_with_timestamps[c2_removed_sentence_indx]['sentence'],
                    'begin_time': abnormality_sentences_with_timestamps[c2_removed_sentence_indx]['begin_time'],
                    'end_time': abnormality_sentences_with_timestamps[c2_removed_sentence_indx]['end_time'],
                }
                curr_dicom_id_perceptual_error_details['class_label_1'] = 1
                curr_dicom_id_perceptual_error_details['class_label_2'] = 1
                curr_dicom_id_perceptual_error_details['missed_fixation_points'] = convert_df_to_dict(
                    c1_c2_missed_fixations_df)
                curr_dicom_id_perceptual_error_details['fixation_points_reduced'] = convert_df_to_dict(
                    c1_c2_fixations_reduced_df)
                curr_dicom_id_perceptual_error_details['phrases_removed'].append(
                    c1_abnormality_removed_sentence_info)
                curr_dicom_id_perceptual_error_details['phrases_removed'].append(
                    c2_abnormality_removed_sentence_info)
            elif subgroup_labels[curr_subgroup_indx] == 'class3':
                c3_fixations_df = create_class3_perceptual_error(
                    fixation_csv_path)
                curr_dicom_id_data['class_label_3_data'] = convert_df_to_dict(
                    c3_fixations_df)
                curr_dicom_id_perceptual_error_details['class_label_3'] = 1
                c3_abnormality_removed_sentence_info = {
                    'class_label_1': 0,
                    'class_label_2': 0,
                    'class_label_3': 1,
                    'phrase': abnormality_sentences_with_timestamps[removed_sentence_indx]['sentence'],
                    'begin_time': abnormality_sentences_with_timestamps[removed_sentence_indx]['begin_time'],
                    'end_time': abnormality_sentences_with_timestamps[removed_sentence_indx]['end_time'],
                }
                curr_dicom_id_perceptual_error_details['phrases_removed'].append(
                    c3_abnormality_removed_sentence_info)

            aggregated_dicom_data[current_dicom_id] = curr_dicom_id_data
            aggregated_dicom_perceptual_error_info[current_dicom_id
                                                   ] = curr_dicom_id_perceptual_error_details

            dicom_id_indx += 1
            print(
                "================================================================================")

        curr_subgroup_indx += 1

    with open('aggregated_dicom_data.json', 'w') as json_file:
        json_file.write(json.dumps(aggregated_dicom_data, indent=4))

    with open('aggregated_dicom_perceptual_error_info.json', 'w') as json_file:
        json_file.write(json.dumps(
            aggregated_dicom_perceptual_error_info, indent=4))

    with open('dicom_abnormality_transcript_timestamps.json', 'w') as json_file:
        json_file.write(json.dumps(
            dicom_abnormality_transcript_timestamps, indent=4))


if __name__ == '__main__':
    main()
