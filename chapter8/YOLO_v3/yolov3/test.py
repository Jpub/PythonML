#import tensorflow as tf

#max_grid_w = 7
#max_grid_h = 7
#batch_size = 64

#cell_x = tf.cast(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)), tf.float32)
#cell_y = tf.transpose(cell_x, (0,2,1,3,4))
#cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])
#print(cell_grid)

#print(1/4)

for i in range(5):
    print(i)

