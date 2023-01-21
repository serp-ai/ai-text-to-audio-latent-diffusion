training_dir - training data directory

batch_size - batch size

num_gpus - number of GPUs to use for training

num_nodes - number of nodes to use for training

num_workers - number of CPU workers for the DataLoader

sample_size - number of audio samples for the training input (uncompressed 24hz)

demo_every - number of steps between demos
     
demo_steps - number of denoising steps for the demos  

num_demos - number of demos to create

ema_decay - the EMA decay      

seed - the random seed

accum_batches - How many batches for gradient accumulation

checkpoint_every - number of steps between checkpoints                            

cache_training_data - if true training data is kept in RAM 

random_crop -  if true randomly crop input audio (for augmentation)

ckpt_path - checkpoint file to (re)start training from (use the ema weights and the non-ema will automatically load with it)

save_path - path to output the model checkpoints

resume_from_checkpoint - resume training from checkpoint

precision - what precision to use for training

lr - learning rate

scale_lr - whether or not to scale the learning rate (lr * accum_batches * batch_size * num_gpus)

lr_warmup_steps - learning rate warmup steps

use_8bit_optim - if true use 8-bit optimizer

gradient_checkpointing - if true use gradient checkpointing

adam_beta1 - adam beta1

adam_beta2 - adam beta2

adam_epsilon - adam eps

adam_weight_decay - adam weight decay

max_grad_norm - max gradient norm

num_epochs - number of epochs until training is finished

max_train_steps - maximum number of training steps until training is finished

lr_scheduler - what learning rate scheduler to use

target_bandwidth - target bandwidth for Encodec's compression

train_text_encoder - if true train the text encoder

embedder_path - path to the text encoder

use_text_dropout - if true use '' in place of the prompt randomly based on the text_dropout_prob

text_dropout_prob - chance that the prompt will be dropped

shuffle_prompts - if true randomly shuffle the prompt by the shuffle_prompts_prob split by the seperator string (shuffle_prompts_sep)

shuffle_prompts_sep - string to seperate prompt by

shuffle_prompts_prob - probability that the prompt will be shuffled