import os
import tensorflow as tf

### this python script will be used to convert the downloaded tensor flow files for bc_z into images and videos.






PATH_TO_DATA = '/media/fabian/Seagate Portable Drive/bc_z/bcz-79task_v16.0.0_failures.tfrecord/'


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# def main():

#     def find_binary_files(path):
#         binary_files = []
#         for root, dirs, files in os.walk(path):
#             for file in files:
#                 print(file)
#                 binary_files.append(os.path.join(root, file))
#         return binary_files

#     binary_files = find_binary_files(PATH_TO_DATA)
#     print(binary_files)
#     def load_binary_file(file_path):
#         raw_dataset = tf.data.TFRecordDataset(file_path)
#         for raw_record in raw_dataset.take(1):
#             example = tf.train.Example()
#             example.ParseFromString(raw_record.numpy())
#             print(example)

#     for binary_file in binary_files:
#         load_binary_file(binary_file
# 
# )


def main():
    def find_binary_files(path):
        binary_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                binary_files.append(os.path.join(root, file))
        return binary_files

    binary_files = find_binary_files(PATH_TO_DATA)
    
    # each of the files are tensorflow datasets
    # for each path in the list load the dataset
    
    def load_binary_file(file_path):
        
        raw_dataset = tf.data.TFRecordDataset(file_path)
        for raw_record in raw_dataset.take(3):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            print(example)

    for binary_file in binary_files:
        load_binary_file(binary_file)
        
    print(binary_files)
        

if __name__ == '__main__':
    main()