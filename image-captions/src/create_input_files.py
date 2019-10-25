from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/raid/IC-GAN/img_captions/coco2014_bottomup_captions/caption_datasets/dataset_coco.json',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/raid/IC-GAN/img_captions/image-captioning-bottom-up-top-down-master/dataset_2014',
                       max_len=50)
