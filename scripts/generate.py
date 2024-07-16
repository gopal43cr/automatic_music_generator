from model import *

# Music generation function
def generate_music(model, x_test, timesteps, ind2note, device):
    model.eval()
    start_idx = np.random.randint(0, len(x_test) - 1)
    input_sequence = x_test[start_idx]
    input_sequence = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)

    generated_notes = []

    for _ in range(200):  # Generate 200 notes
        with torch.no_grad():
            predictions = model(input_sequence)
            predicted_note_idx = torch.argmax(predictions, dim=-1).item()

        predicted_note = ind2note[predicted_note_idx]
        generated_notes.append(predicted_note)

        input_sequence = torch.cat([input_sequence[:, 1:], torch.tensor([[predicted_note_idx]], dtype=torch.long).to(device)], dim=1)

    return generated_notes

# Generate music
generated_notes = generate_music(model, x_test, timesteps, ind2note, device)
