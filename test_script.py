################################################################################
# filename: test_script.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 15/05,2025
################################################################################

from models.decisions_models.game_state_encoder import GameStateEncoder
from models.decisions_models.mlp_yugioh import YuGiOhMLP
from models.BoardGame import GameBoard
import torch
import torch.nn.functional as F

################################################################################

encoder = GameStateEncoder(vocab_size=5000, embed_dim=128)
model = YuGiOhMLP(input_dim=1024, hidden_dim=512, num_card_ids=5000)

boardGame = GameBoard(
    self_main_ids=[101, 222, 345],
    opp_main=2,
    self_field_ids=[3001],
    opp_fields_ids=[4001],
    self_graveyard_ids=[],
    self_banish_ids=[],
    self_banish_verso=0,
    opp_fields_verso_card=1,
    opp_graveyard_ids=[],
    opp_banish_ids=[],
    opp_banish_verso=0,
    phase_id=3,
    lp=6000,
    adv_lp=4500,
)

# Fake input
vec = encoder(**boardGame.get_dict())

card_logits, zone_logits, action_logits = model(vec.unsqueeze(0))  # (batch_size=1)

card_probs = F.softmax(card_logits, dim=-1)
zone_probs = F.softmax(zone_logits, dim=-1)
action_probs = F.softmax(action_logits, dim=-1)

card_id = torch.argmax(card_probs, dim=-1).item()
zone_id = torch.argmax(zone_probs, dim=-1).item()
action_id = torch.argmax(action_probs, dim=-1).item()

print(f"🔮 Carte proposée : ID {card_id}")
print(f"📦 Zone ciblée   : {zone_id}")
print(f"🎬 Action choisie: {action_id}")

################################################################################
# End of File
################################################################################