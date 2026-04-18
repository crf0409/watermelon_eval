from gradio_client import Client, handle_file

client = Client("http://10.4.64.36:124/")
result = client.predict(
	image=handle_file('/home/siton02/md0/crf/watermelon_eval/dataset/19_datasets/2_9.7/chu/1/1.jpg'),
	audio=handle_file('/home/siton02/md0/crf/watermelon_eval/dataset/19_datasets/2_9.7/chu/1/1.wav'),
	true_brix=None,
	api_name="/predict_watermelon"
)
print(result)