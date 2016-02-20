import glob
import sys

def get_value(s):
    return float(s.split()[1])

def process_file(fname):
    try:
        lines = [l for l in open(fname)]
        idx = None
        args = None
        for i in xrange(len(lines)):
            if lines[i].find('test_perf') > -1:
                idx = i
            elif lines[i].find('encoder') > -1:
                args = lines[i]
        return {
            'fname': fname,
            'args': args,
            'val_r1@10': get_value(lines[idx-3]),
            'test_r1@2': get_value(lines[idx+2]),
            'test_r1@10': get_value(lines[idx+4]),
            'test_r2@10': get_value(lines[idx+5]),
            'test_r5@10': get_value(lines[idx+6])
        }
    except:
        print 'Error reading:', fname
        return None

pattern = sys.argv[1]
print 'pattern:', pattern
fnames = glob.glob(pattern)
print 'fnames:', fnames
best_result = None
for fname in fnames:
    curr_result = process_file(fname)
    if not curr_result:
        continue
    if best_result is None or best_result['val_r1@10'] < curr_result['val_r1@10']:
        best_result = curr_result
print best_result
