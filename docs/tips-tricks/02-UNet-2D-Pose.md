# Tips and Tricks
### Specific details required to get this version to work, both common and crazy

## Batch Overload
AUC metric caused a lot of OOM errors that I couldnt track down at first. Seems like that metric just uses a lot of memory, and the batch size must be dramatically decreased in order to use it. So, I just used precision and recall instead and only decreased the batch size a little.

## Memory (RAM and GPU)
**Bottom Line: Just use data generators.** I thought loading the full dataset would be faster in bulk training, and with AI platform notebooks you have access to some huge machines that can do it. However, it really doesnt seem that much faster, and is so much harder to configure.

Using a data generator manages that for you, and just runs at max RAM and GPU. The only overload will come from a batch overload. However, using a data generator means that everything must flow through the model.fit_generator data (no extra callback data in memory). Otherwise tf will trying to maximize memory using and cause OOM.

## Custom Metrics and Losses
Spent a lot of time writing custom Losses and Metrics, especially a custom metric (called NormKPM) that meant to find the average distance from predicted and real joints in pixel scale. **Using non-max suppression to get the joint coordinates seems like the best solution.** However, the real trouble is the variable number of joints in each heatmap. With a custom loss/metrics, you've got to write it in tensor functions (no looping, no numpy) and doing math on variable length tensors is something that I did not solve. Also, tf.where to find the coordinates of joints in a heatmap (after non-max suppression) rearranges the data in a really tricky way to do anything with after - it makes a 2D tensor without any flexibility.

So, I tried to use a numpy implementation, looping over heatmaps to recover a metric that was pretty much the distance from predicted to actual joint coordinates. I used this function in a callback, where tensor logic is not required, but things got really tricky with memory usage - first, validation data for this calculation had to be fully loaded because it can't (for the current version of tf/keras) be accessed from a callback. This caused a lot of memory problems, especially when using generators.

**At the end of the day, if existing losses/metrics work, dont go crazy trying to implement a cool loss/metric.**