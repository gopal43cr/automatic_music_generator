from load_data import *
# Unique notes and frequency calculation
notess = sum(notes_array, [])
unique_notes = list(set(notess))
print("Unique Notes:", len(unique_notes))
thresold = 50
freq = dict(map(lambda x: (x, notess.count(x)), unique_notes))
freq_notes = dict(filter(lambda x: x[1] >= thresold, freq.items()))

if not freq_notes:
    raise ValueError("No notes found with frequency >= 50.")

new_notes = [[i for i in j if i in freq_notes] for j in notes_array]
ind2note = dict(enumerate(freq_notes))
note2ind = dict(map(reversed, ind2note.items()))

timesteps = 50
x, y = [], []

for i in new_notes:
    for j in range(0, len(i) - timesteps):
        inp = i[j:j + timesteps]
        out = i[j + timesteps]
        x.append(list(map(lambda x: note2ind[x], inp)))
        y.append(note2ind[out])

x_new = np.array(x)
y_new = np.array(y).reshape(-1,)

x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, random_state=42)