from pylsl import StreamInlet, resolve_stream, resolve_bypred
#from pylsl import StreamInfo, StreamOutlet, resolve_stream

print("looking for an EEG stream...")
#streams = resolve_stream('name', 'AURA')
#streams = resolve_stream('name', 'AURA_Filtered')
streams = resolve_stream('name', 'AURA_Power')



inlet = StreamInlet(streams[0])
while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    sample, timestamp = inlet.pull_sample()
    print(timestamp, sample)