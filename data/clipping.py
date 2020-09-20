from np_sound import NPSound

import os

for root, dirs, files in os.walk("./piano/"):
    for x in files:
        if x.endswith(".wav"):
            wav_file = NPSound(os.path.join(root, x))
            wav_file.clip_at_threshold(100).write("piano_clipped/" + x)
