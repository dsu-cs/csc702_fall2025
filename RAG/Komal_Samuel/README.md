# RAG Groupwork Project - Comparing Chunk Sizes

For our project for this groupwork phase, we decided to see how different chunk sizes affect generated responses with a pretrained LLM with RAG. We first tried to download a model locally on the VM from Meta's LLaMa, however we were not able to get the necessary libraries working in order to properly utilize the LLM. We then moved onto remotely accessing a LLaMa model by using Ollama, this worked out much better as the pipeline was much easier to put together and get working. For the RAG portion of the project, we used LlamaIndex and their many different classes to take in RAG documents and index them. Wikipedia API was used to pull articles to use as RAG documents and store them in a 'docs/' directory.

For the testing of the different chunk sizes, we decided on the sizes of 128, 256, 512, and 1024. The number of the chunk size refers to the number of tokens in each chunk, or close to it as the SentenceSplitter we used tries to keep full sentences intact. So for each size, we loaded in the documents and used the SentenceSplitter to chunk the data into the specified size, then we put them into a retrieval index so Ollama can pull them correctly. In order for the LLM to pull the correct chunks, we used the all-MiniLM-L6-v2 embedding model so the models is able to see which chunks are most related to the query. The article that we pulled from Wikipedia are all about different animals, so we asked the LLM about certain animals in the documents. The results from the different queries can be seen at the end of this document as they are quite long. After our prompting we also tried to test retrieval accuracy with the different chunks sizes, however all of the chunk sizes got 100% accuracy, so we did not gain any insight from it.

The results that we saw from our testing we pretty intesting. The answers looked quite different based on the chunk size the LLM was using for its RAG. The biggest difference is in the first query where one chunk size says the complete opposite of all others, saying that lions do not hunt together and are solitary creatures, while the rest say they do hunt together. Another big difference is the size of each response. Even though all responses were given the same limit of 256 output tokens, some chunk sizes produced short answers while other produced lengthly ones. From what can gathered from the results, it looks like the chunk sizes of 128 and 1024 produced the shortest answers. Although the responses do look different, much of the underlying information is the same, except for the query about lions. An example of this is when we asked 'What do elephants eat?' and all of the responses included very similar lists, or when different chunk sizes mentioned the same number for a cheetah's top speed.


# Results

Query: Explain how lions hunt together.

=== Chunk size: 128 ===
Response: Lions are known to hunt together as a coordinated effort, often relying on their senses of sight and hearing to locate prey. When they have located potential prey, they will stalk and surround it, working together to corner and subdue it. This collective hunting behavior is an important aspect of lion social dynamics and plays a crucial role in their survival and success as a species.

=== Chunk size: 256 ===
Response: Lions do not have a hunting strategy that involves them working together in the same way that tigers do. While lions are also ambush predators, they typically hunt alone or in small groups, rather than as part of a larger family unit. In fact, lions are known to be relatively solitary animals, with males often living and hunting alone outside of the breeding season. During this time, males will come together to mate with females, but afterward, they will go their separate ways until the next mating season.

Lions are skilled hunters and use a variety of tactics to catch their prey. They are known for their stealth and speed, and can move quietly and quickly over long distances in pursuit of their quarry. Once they have identified potential prey, they will stalk it silently and then make a sudden charge when they are close enough to make a kill. This type of hunting is often referred to as "stalking and pounce."

While lions do not hunt together in the same way that tigers do, they are still highly effective predators and play a crucial role in their ecosystems. Their social structures and hunting strategies have evolved over time to ensure their survival and success in their environments.

=== Chunk size: 512 ===
Response: Lions are known to be social predators, meaning they often hunt together in groups or coalitions. This cooperative hunting behavior is beneficial for several reasons. Firstly, it allows them to take down larger and more formidable prey than they would be able to tackle alone. Secondly, it provides an opportunity for younger lions to learn the skills and strategies of hunting from their elders. Finally, it enables them to share the spoils of the hunt and reduce competition among each other.

When lions hunt together, they typically follow a coordinated approach. They use their keen senses of smell and hearing to locate potential prey, and then work together to stalk and surround it. Once they have the prey cornered, one or more lions will make a sudden charge towards it, using their powerful legs and sharp claws to bring it down.

In some cases, lions may also use a strategy called "hunting by driving." In this approach, several lions work together to drive the prey towards a central location, where they can more easily attack and kill it. This tactic is often used when hunting larger and more elusive prey, such as buffalo or rhino.

Overall, the ability of lions to hunt together is an important adaptation that has helped them become successful predators in their ecosystem. By working together, they are able to take down prey that would be too difficult or dangerous for a single lion to handle alone.

=== Chunk size: 1024 ===
Response: Lions are known to hunt together in coordinated manner, with each member of the pride playing a crucial role in the successful kill. The lions use a variety of tactics to hunt together, including:

1. Stalking: Several lions will stalk their prey simultaneously, using stealth and concealment to get close enough for a kill.
2. Ambush: Lions will often set up an ambush around a watering hole or other area where their prey is likely to pass through. When the prey enters the ambush zone, the lions will spring into action and surround it.
3. Choke point: Lions may also use a choke point tactic, where they converge on a narrow passage or corridor where their prey must pass through. Once the prey is funneled into the choke point, the lions will attack.
4. Coordinated attacks: Once the lions have surrounded and cornered their prey, they will use coordinated attacks to bring it down. This may involve multiple lions biting and clawing at the same time, or one lion taking the lead in a synchronized attack.

Overall, lions are highly social animals that have evolved to hunt together in a coordinated manner. Their hunting tactics are often more effective when they work together, as they can cover more ground and take down prey much larger than themselves.



Query: What do elephants eat?

=== Chunk size: 128 ===
Response: Elephants are herbivorous and have a diverse diet that consists of various types of leaves, twigs, fruit, bark, grass, and roots. According to the context information provided, African elephants mostly browse, while Asian elephants mainly graze. Additionally, it is mentioned that elephants can eat as much as 300 kg (660 lb) of food and drink 40 L (11 US gal) of water in a day, indicating their ability to consume large amounts of food.

=== Chunk size: 256 ===
Response: Elephants are herbivorous and their diet consists of a variety of plants, including leaves, twigs, fruit, bark, grass, and roots. They can eat up to 300 kg (660 lb) of food and drink 40 L (11 US gal) of water in a day.

=== Chunk size: 512 ===
Response: Elephants are herbivores, which means they primarily feed on plant-based food sources. Their diet typically consists of a wide variety of vegetation, including grasses, leaves, fruits, and bark. In the wild, elephants have been observed eating over 100 different types of plants, making them one of the most diverse herbivores in the animal kingdom.

In addition to their plant-based diet, elephants may also consume small amounts of insects and other invertebrates, such as ants and termites. However, this is not a significant part of their diet and is largely incidental.

It's worth noting that the specific dietary preferences of elephants can vary depending on their age, sex, and geographic location. For example, African elephants tend to favor more fibrous plant material, such as shrubs and tree bark, while Asian elephants tend to prefer softer, more palatable foods like grasses and leaves.

=== Chunk size: 1024 ===
Response: Elephants are herbivorous and they mainly feed on vegetation such as grasses, leaves, fruits, and bark. They also consume water and mud to help them regulate their body temperature and protect their skin from the sun. In addition, elephants have been known to eat seeds, especially in African forest elephant populations, where they play a crucial role in seed dispersal.



Query: Describe the habitat of penguins.

=== Chunk size: 128 ===
Response: Penguins inhabit a variety of aquatic environments, including the Antarctic and sub-Antarctic regions, as well as parts of South America, Africa, Australia, and New Zealand. They are found in areas with cold temperate and tropical climates, and their habitats range from rocky coastlines to sandy beaches, and even ice sheets. These areas provide the necessary conditions for penguins to thrive, including access to food sources such as fish, krill, and squid, as well as suitable nesting sites.

=== Chunk size: 256 ===
Response: Penguins are semi-aquatic birds that inhabit a variety of habitats, primarily found in the Southern Hemisphere. Their natural habitats include cold climates, such as Antarctica and the surrounding islands, as well as temperate regions along coastlines. Penguins can be found in both land and sea environments, with some species living exclusively in the water and others spending half of their time on land and half in the sea.

The emperor penguin, being the largest species, is known to inhabit the icy waters of Antarctica and surrounding areas. They can dive to depths of approximately 550 meters while searching for food, and their thick layer of insulating feathers helps keep them warm in the cold water.

Other penguin species, such as the Gentoo and Chinstrap penguins, can be found in the Antarctic Peninsula and surrounding islands. These species are also adapted to life in cold climates, but they are more flexible in their habitat preferences than the emperor penguin.

In addition to Antarctica, penguins can also be found in temperate regions along coastlines. For example, the Rockhopper penguin is found in the southern parts of South America, while the Macaroni penguin is found in the southern parts of Africa. These species are adapted to life in warmer climates and are more tolerant of saltwater than the emperor penguin.

Overall, penguins inhabit a variety of habitats, ranging from cold climates to temperate coastal regions. Their adaptability to different environments allows them to thrive in a wide range of ecosystems.

=== Chunk size: 512 ===
Response: Penguins are found in a variety of habitats around the world, including cold climates, temperate regions, and even tropical areas. However, their natural habitat is primarily in the Southern Hemisphere, where they can be found living in Antarctica and the surrounding islands, as well as in the southern parts of South America, Africa, Australia, and New Zealand.

In these regions, penguins are adapted to live in cold water and on land, with some species spending more time in the water than others. They are highly skilled swimmers and divers, able to dive to great depths in search of food. On land, they use their flippers to waddle or slide on their bellies, conserving energy while moving quickly.

Penguins are also known to migrate seasonally in search of food and more favorable living conditions. For example, some species may migrate to warmer waters during the winter months to escape harsh weather conditions and find food. Others may stay in their year-round habitats and adapt to changes in temperature and sea ice.

Overall, penguins are highly adapted to their aquatic and terrestrial environments, and their habitats are characterized by cold temperatures, rocky or icy terrain, and access to water for swimming and diving.

=== Chunk size: 1024 ===
Response: Penguins are found in habitats in the Southern Hemisphere, primarily in Antarctica and surrounding islands, as well as in the southern parts of South America, Africa, and Australia. They live almost exclusively in the water and on land, with only a few species inhabiting colder regions. The largest living species, the emperor penguin, breeds at temperate waters off Antarctica, while smaller species can be found in warmer climates. Penguins have adapted to life in the ocean water, with counter-shaded dark and white plumage and flippers for swimming. They spend about half of their lives on land and the other half in the sea.



Query: How fast can a cheetah run?

=== Chunk size: 128 ===
Response: Cheetahs are capable of reaching incredible speeds, with some studies suggesting they can reach up to 104 km/h (65 mph) over short distances. However, it's important to note that these high speeds are not sustainable for long periods of time and are typically achieved during short bursts of intense activity, such as during hunting or escape attempts.

In addition to their impressive top speed, cheetahs are also known for their agility and quick reflexes, which allow them to accelerate rapidly and change direction with ease. This combination of speed and agility makes cheetahs some of the fastest land animals on Earth.

=== Chunk size: 256 ===
Response: Based on the provided context information, the maximum speed recorded for a cheetah is approximately 100.1 km/h (58 mph) sustained for 1-2 seconds, as mentioned in the text.

=== Chunk size: 512 ===
Response: According to the provided context information, the top speed of a cheetah has been recorded at 100.1 km/h (62.1 mph) for one individual. Additionally, other studies have measured the running speed of cheetahs as follows:

* Average running speed of cheetahs during hunts is estimated to be around 58 mph (93 km/h) based on GPS-IMU collars.
* The maximum sustained speed of cheetahs has been recorded at 104 km/h (65 mph) over a marked course.

Therefore, the fastest speed that a cheetah can run is approximately 100.1 km/h (62.1 mph), but they can sustain speeds of up to 104 km/h (65 mph) in short bursts.

=== Chunk size: 1024 ===
Response: Based on the provided context information, the fastest speed of a cheetah has been measured at approximately 104 km/h (65 mph) over a marked 201 m (220 yd) course. This was achieved through three runs, starting from behind a line and including measurements for both maximum sustained speed and running speed. The information provided in the context suggests that the cheetah is capable of reaching incredible speeds, with estimates suggesting it can run up to 150 km/h (93 mph) for short distances. However, it's important to note that these measurements were taken in controlled conditions and may not reflect the cheetah's actual running speed in real-world scenarios.