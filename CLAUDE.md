# Objective
We aim to create a multi-agent reinforcement learning project where each agent represents a character that is playable in the game. We want to find optimal policies for each character. 

## Goals
1. A reinforcement learning environment and mapping of the board game mechanics to Markov Decision Process (MDP)
2. A simulation with multiple agents trying to learn the best policy for the given character
3. Character's global win rates and probabily to win a random game
4. Character's win rates and probability to win based on the player pool
5. Character matchups e.g. probability to win if another character is also playing
6. Summarized strategies for each character both globally and match-up based
7. Analysis on the differences between RL methods

I will provide more details and information about the game that we will relate to the Markov Decision Process and our RL project

# Primary Rules Summary
Objective is to score the most points, i.e. win the most votes, in total by winning the most delegates from regional elections. To do this, you must have the most influence cubes in a region.Influence cubes are gained and deployed by using different Action cards. Each round, players set four Action cards face down and are commited to executing the cards in order. That is, the player who starts the round will execute their first action card, then the next player will execute their first action card, and so forth until all players have executed their action cards. Once everyone is finished executing their actions, the active voting regions for that round are assessed for influence cubes and the number of votes (points) are awarded to the winner or winners. The game ends after the the final round, round eleven, is finished and then the candidate with the most votes (points) wins.

## Rounds

There are eleven rounds with rounds one, four, and nine being non voting rounds. The remainder of the rounds are assigned regions before the game starts and is known to all players. Here is a map of rounds: number of regions eligible for voting
{
    "One": 0
    "Two": 1,
    "Three" 2,
    "Four": 0,
    "Five" 1,
    "Six": 2,
    "Seven": 1,
    "Eight": 2,
    "Nine": 0,
    "Ten": 2,
    "Eleven": 3
}


## Initial Conditions
The starting player is randomly assigned before the game and then each round the starting player shifts clockwise. We can just assign a set sequence of ordering before the game starts. The first players chooses a starting "home region" for their character token and places two Influence cubes from their bank on that region. Each player card has a different amount of initial Influence cubes in their hands

## News
- At the beginning of every round, first player draws a News card and all players resolve its effect
- News card effects apply to the current round unless stated otherwise
- News card effects apply to all players, not just the player reading it
- When applicable, execute News card effects in turn order, starting with the first player

## Action
- Each player commits 4 Action cards in the order of execution face-down so it is unknown to other players
- Each player takes turns executing their leftmost Action card
- The Super Pac action card is the only Action card whose effect is awarded at the end of the round
- All players have access to the same 14 Action cards unless determined otherwise by News effects

## Voting
- Some regions are winner take all so the candidate with the most Influence cubes will get all the votes
- Some regions have 2nd place awards given to the candidate with the second most Influence cubes
- If there is only 1 player with Influence on a region with two awards, then the player claims the entire point total
- If there is a tie on a region with two awards, the total point total is evenly split to all tied players and rounded down to the nearest whole number

## Win Conditions
- The player with the most points at the end of round 11 wins
- First tie breaker, the player with the most influence cubes in their hand wins
- Secondary tie breaker, the player with the most influence cubes in the bank wins
- If all tie breakers are exhausted and there is a tie, then we will call the game a draw. We want to track how often this happens

## Influence Cubes

- Each player has 30 influence cubes
- Influence cubes in the players hand are active and can be placed on a voting region
- Incluence cubes in the bank are not active for placement unless there is a reason otherwise
- At the end of each round, after voting is resolved, the influence cubes placed on regions that were eligible for voting are returned to the players banks
- We can think of Influence cubes as the game currency

## Phases Summary
1. News
2. Actions
3. Voting

# Primary Game Map Environment
The game map is a map of the United States broken out into 14 regions with various point offerings. Note Alaska and Hawaii does not share borders with any other region and are looped in with the West region. The regions are what's important not the states. I will describe which regions share borders.

## Region to US State mapping
{
    "Pacific NW": ["Washington", "Oregon, "Idaho"],
    "West": ["California", "Nevada", "Alaska", "Hawaii"],
    "South West": ["Colorado", "Utah", "Arizona", "New Mexico"],
    "Greater Texas": ["Texas", "Oklahoma"],
    "North": ["Montana", "Wyoming", "North Dakota", "South Dakota"],
    "Great Lakes": ["Minnesota", "Wisconsin", "Michigan"],
    "Early Midwest" ["Illinois", "Ohio", "Indiana"],
    "Midwest": ["Nebraska", "Iowa", "Missouri", "Kansas"],
    "Deep South": ["Lousiana", "Mississippi", "Arkansas"],
    "South East": ["Florida", "Alabama", "Georgia"],
    "South": ["Tennessee", "North Carolina", "South Carolina"],
    "Appalachia": ["Kentucky", "West Virgina", "Virginia"],
    "North East": ["Pennsylvania", "New Jersey", "Maryland", "Delaware"],
    "New England": ["New York", "Vermont", "New Hampshire", "Massachusetts", "Connecticut", "Maine", "Rhode Island"]
}

## Regions sharing borders with other regions
{
    "Pacific NW": ["North", "South West", "West"],
    "West": ["South West", "Pacific NW"],
    "South West": ["North", "Pacific NW", "West", "Midwest", "Greater Texas"],
    "Greater Texas": ["South West", "Midwest", "Deep South"],
    "North": ["Pacific NW", "South West", "Midwest", "Great Lakes"],
    "Great Lakes": ["North", "Midwest", "Early Midwest"],
    "Early Midwest" ["Midwest", "Great Lakes", "North East", "Appalachia"],
    "Midwest": ["Early Midwest", "North", "South West", "Greater Texas", "Deep South", "Great Lakes", "Appalachia", "South"],
    "Deep South": ["Midwest", "Greater Texas", "South East", "South"],
    "South East": ["Deep South", "South"],
    "South": ["Midwest", "Deep South", "South East", "Appalachia"],
    "Appalachia": ["Midwest", "South", "Early Midwest", "North East"],
    "North East": ["New England", "Early Midwest", "Appalachia"],
    "New England": ["North East"]
}

## Points for Each region
Mapping of regions to points
{
    "Pacific NW": [13],
    "West": [17, 7],
    "South West": [11, 5],
    "Greater Texas": [15, 6],
    "North": [10],
    "Great Lakes": [14],
    "Early Midwest" [13, 5],
    "Midwest": [15],
    "Deep South": [11],
    "South East": [16, 7],
    "South": [12, 5],
    "Appalachia": [12],
    "North East": [14, 6],
    "New England": [18, 8]
}

# News Cards
News cards affect actions each player can do on that turn. News cards are read prior to the round starting. I will provide a mapping of card name and card description. News cards are selected randomly without replacement. We need to codify the effects of the news cards. If a News card says "may" or similarly choiced words, that indicates an optional choice in the respective action. If a News card says "must" or similar, that indicates the action is necessary

{
    "Countermeasures": "All players gain 1 influence cube form the bank each time they enter a region where any opponent has Influence",
    "Oppo Research": "All players who end this round in the same region as any opponent gain 2 Influence cubes from the bank",
    "Generous Donors": "All players gain 1 extra Influence cube for each Fundraiser card they play",
    "Tax Reform": All players immediately gain 2 Influence cubes from the bank,
    "Last Minute Debate": "All players may immediately move to a region voting this round or next round. This does not cost an Influence cube",
    "Detour": "All players must immediately move to a new region. This does not cost an Influence cube.",
    "Frozen Assets": "Action cards played this round cannot be used next round. If this is drawn during round 11, disregard it and draw another News card",
    "Paparazzi": "Bus an Plane cards allow all players to place 1 Influnce cube from the bank onto their destination region",
    "Spin Doctor": "Positive Ad and Negative Ad cards may be used as either card type",
    "Final Push": "All players may play 5 Action cards",
    "Long Term Planning": "All players may immediately move to a region voting in round 11. This does not cost an Influence cube",
    "Deregulation": "Whoever plays the most Fundraiser cards gains 2 extra Influence cubes. If there is a non-zero tie, all tied players gain 1"
    "Extra Airtime": "Positive Ad cards allow players to place 2 Influence cubes from their hands onto any region(s)",
    "Wall Street Support": "The first Rally card played by each player uses 2 Influnce cubes from the bank, instead of from their hands",
    "Road Trip": "Bus cards allow players to move 1 adjacent region further",
    "Plutocracy": "Whoever plays the most Super PAC cards gains 6 Influence cubes. If there is a non-zero tie, all tied players gain 3",
    "New Constituents": "All players may immediately swap the location of their pawn and any Influence cube on the map",
    "Frequent Flyer": "Plane cards do not cost an Infuence cube"
}


# Action Cards
Mapping of action cards to number of cards and card effect.

{
    "Super PAC": [4, "If you play the most Super PAC cards, gain 4 Influnce cubes. If tied, gain 2],
    "Fundraiser": [3, "Gain 1 Influence cube"],
    "Plane": [1, "Pay 1 Influence cube and move to any region"],
    "Bus": [1, "Move to any adjacent region"],
    "Positive Ad": [1, "Place 1 Influence cube onto any region"],
    "Negative Ad": [1, "Remove any 1 Influence cube from any region"],
    "Rally": [3, "Place 2 Influence cubes onto your current region"]
}

Note that News cards can change the rules and effects of Action cards.

# Candidate Information and Abilities
Candidate mapping to candidate special ability and number of initial influence cubes
{
    "Antonio Parades": ["Bus cards allow you to move up to 2 adjacent regions away from your current location", 5],
    "Kenneth Walker": ["If you play 2 or more Fundraiser cards in a round, gain 1 extra Influence cube from the bank", 2],
    "Bryce McKay": ["After each vote where you do not win any delegates, gain 1 Influence cube from the bank", 3],
    "Kristen Ramirez": ["If you play the most Super PAC cards in a round, gain 1 extra Influence cube from the bank", 3],
    "Diane Proudian": ["Each round you may choose to not execute 1 Action card in your queue", 3],
    "Ryan Gupta": ["Plane cards do not cost you an Influence cube", 1],
    "Chester Freeman": ["If you end a round in the same region as any opponent, gain 1 Influence cube from the bank", 3],
    "Lily Nguyen": ["The first Rally card you play each round may place 1 extra Influence cube from your hand onto your current region", 2],
    "Erin Fields": ["Before executing your first Action card each round, you may move 1 of your Influence cubes on the map to any region", 5],
    "Calvin Matsui": ["Bus cards allow you to place 1 Influence cube from your hand onto your destination region", 3],
    "Anna Patel": ["Your character token counts as 2 Influence cubes in its current region", 3],
    "Georgia Barnes": ["Start the game with 3 home regions and place 2 Influence cubes from the bank in each", 1] 
}

# Markov Decision Process
- Map the components and descriptions of the game to the Markov Decision Process
- Map how different News cards and Candidate abilities affect the available actions, rewards, and game environment
- Create a simple game environment and simulator

# RL Algorithms
- Consider the physical limitations of this laptop and runtime. We may have to run experiments and simulations on a different machine
- Use well-documented and mainstream RL libraries e.g. RLlib or similar
- Try PPO, NFSP, and Dueling DQN. 
- Use your reasoning to select a RL method that I did not list and compare it. Try to beat the other 3

# Evaluations
- Add evaluations and analysis in Jupyter Notebooks with visualizations

# Data Collection
Make sure to store data and results for analysis. Including but not limited to:
- Locations of characters per turn including starting, ending, and in between locations when applicable
- Influence cubes in the hand before and after turn
- Action cards played in turn
- Voting round won or lost or not participated
- Locations of Influence cubes placed or removed

