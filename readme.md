## prepare data

# crop face from picture with scale of 2.5 and square shape
python -m tfFaceCropFromVideo -v /media/macul/black/spoof_db/spoofing_data_Mar_2019/real2 -o /media/macul/black/spoof_db/spoofing_data_Mar_2019/real2_crop25 -sc 2.5 -sz 50 -sq 1
python -m tfFaceCropFromPic -f /media/macul/black/spoof_db/collected/original/image_attack/printed -o /media/macul/black/spoof_db/collected_image_attack_crop25_mtcnn -sc 2.5 -sz 50 -sq 1

# move all pictures to the respective folder such as, the subfolder name should match the label.txt
./train_1/real
./train_1/image_attack
./train_1/video_attack
./test_1/real
./test_1/image_attack
./test_1/video_attack

# modify build_tfrecord_from_image.py then create record file
tf.app.flags.DEFINE_string('train_directory', './train1/',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', './test1/',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', './tfrecord1/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 5,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 5,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 5,
                            'Number of threads to preprocess the images.')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.
tf.app.flags.DEFINE_string('labels_file', './label.txt', 'Labels file')

# run build_tfrecord_from_image.py
source ~/tf_venv/bin/activate
python -m build_tfrecord_from_image

# train the net, first change ./config/net_config_ld_conv.py accordingly
python -m train_net

# save to pb
python -m save_pb -s /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2 -c 7

## test data with checkpoints
python -m spoof_eval_video -c 7 -mf /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2 -v /home/macul/Screencast_2019-05-06_10-28-48.mp4  -t 0.95 -tf 0.96 -um 1 -cs 2.0
python -m spoof_eval_pic -mf /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2 -f /media/macul/black/spoof_db/MSU_MFSD_collected/original/MSU_MFSD_negative/ -t 0.9 -tf 0.95 -pkl /home/macul/dgx_MSU_neg_2_7_mtcnn20_pb.pkl -um 1 -cs 2.0 -c 7
python -m spoof_test_ld_cm1 -mf /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2  -v /home/macul/Screencast_2019-05-06_10-28-48.mp4  -t 0.95 -tf 0.96 -um 1 -cs 2.0 -tfl 1.0 -c 7

## test data with pb file
python -m spoof_eval_video -mf /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2 -v /home/macul/Screencast_2019-05-06_10-28-48.mp4  -t 0.95 -tf 0.96 -um 1 -cs 2.0 -mpb /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2/train_2_7.pb
python -m spoof_eval_pic -mf /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2 -f /media/macul/black/spoof_db/MSU_MFSD_collected/original/MSU_MFSD_negative/ -t 0.9 -tf 0.95 -pkl /home/macul/dgx_MSU_neg_2_7_mtcnn20_pb.pkl -um 1 -cs 2.0 -mpb /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2/train_2_7.pb
python -m spoof_test_ld_cm1 -mf /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2  -v /home/macul/Screencast_2019-05-06_10-28-48.mp4  -t 0.95 -tf 0.96 -um 1 -cs 2.0 -tfl 1.0 -mpb /home/macul/libraries/mk_utils/tf_spoof/dgxout/train_2/train_2_7.pb

