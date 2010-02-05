=============================
Experiment Planning
=============================

Incorporating Findings into the Model of Models
-----------------------------------------------

Robomendel has discovered that a phenotypic model for the color of the flowers on a pea plant fits his observations well, but has an apparent contradiction his model of models as to how pea plants behave. How is it that there are pea plants of two different colors? Do the different colors indicate that there are two different species of plants in robomendels virtual garden or that there is a further underlying mechanism for the behavior of the observed characteristic?

To test the species model, Robomendel decides to peform two experiments in the next growing season: a Wh x Wh cross and a Wh x Pu cross. Robomendel selects a plant with white flowers and a plant with purple flowers:

   >>> from darwin.robomendel import plantWh, plantPu, determine_color, multiset, Multinomial

Crossing these plants should shed light on the same species versus different species hypothesis. Robomendel checks the results from cross the white plant with itself:

    >>> from darwin.robomendel import PeaPlant, determine_color, multiset
    >>> white_plant = PeaPlant(genome=PeaPlant.white_genome)
    >>> white_crosses = [white_plant * white_plant for i in range(10)]
    >>> print multiset([determine_color(x) for x in white_crosses])
    {'white': 10}

Looks like all white-flowered progeny! His observations are in line with the predictions for the different species model.



Robomendel also crosses plantWh and plantPu. Robomendel checks the results.

    >>> purple_plant = PeaPlant(genome=PeaPlant.purple_genome)
    >>> hybrid_crosses = [white_plant * purple_plant for i in range(10)]
    >>> print multiset([determine_color(x) for x in hybrid_crosses])
    {'purple': 10}

Not only did the seeds produce plants, all the flowers are purple! This fits the prediction of the same species model! Not only has RoboMendel failed to result the source of the white flowers observation, he has another contradiction!

What's a robotic scientist to do!? Robomendel knows that in some cases infertile offspring can be produced by different species, so he decides to determine if the hybrid plants are sterile. He crosses a large number of hybrid plants and checks the results.

    >>> import random
    >>> hybrid_crosses = [white_plant * purple_plant for i in range(100)]
    >>> hybrid_progeny = [random.choice(hybrid_crosses) * random.choice(hybrid_crosses) for i in range(100)]
    >>> print multiset([determine_color(x) for x in hybrid_progeny])
    {'purple': 74, 'white': 26}

Not only are the seeds viable, over 25% of the plants have white flowers and nearly 75% have purple flowers! Repeating the experiment, RoboMendel finds that on average 25% of the plants have white flowers.

RoboMendel's latest experiment deals a decisive blow to the different species model, but the pattern of offspring does not fit the bio-object model RoboMendel has been using for his pea plants. RoboMendel can construct a Markov model for his observations, with two states Pu and Wh and transition probabilities Pr(Wh|Pu) = \lambda and Pr(Pu|Pu) = 1 - \lambda . Based on past observations, Robomendel expects 0.25 as the value for lambda.

An assumption underlying this model is that \lambda is equal for all individuals. To test this hypothesis, RoboMendel must track the progeny of each parent rather than performing population wide statistics. First he identifies all the purple-flowered plants from the hybrid cross population.

    >>> purple_plants = [x for x in hybrid_progeny if determine_color(x) == "purple"]

For each purple plant, RoboMendel can cross it with a random sample of other purple plants and examine the progeny.

    >>> from darwin.robomendel import multiset, determine_color
    >>> progeny_counts = list()
    >>> for i in range(0, len(purple_plants)):
    ...     progeny = [purple_plants[i] * random.choice(purple_plants) for j in range(0, 20)]
    ...     progeny_colors = multiset([determine_color(x) for x in progeny])
            if 'white' not in progeny_colors:
                progeny_colors['white'] = 0
    ...     progeny_counts.append(progeny_colors)

Now Robomendel can obtain a simple estimate of \lambda for each plant:

>>> for counts in progeny_counts:
...     print float(counts['white']) / float(counts['white'] + counts['purple']),
...
0.05 0.0 0.1 0.0 0.0 0.2 0.3 0.15 0.15 0.3 0.2 0.15 0.05 0.1 0.15 0.2 0.4 0.1 0.15 0.0 0.0 0.0 0.2 0.0 0.2 0.0 0.0 0.1 0.1 0.25 0.2 0.15 0.35 0.0 0.0 0.15 0.25 0.0 0.3 0.1 0.05 0.0 0.05 0.05 0.0 0.15 0.25 0.0 0.3 0.0 0.3 0.0 0.2 0.35 0.15 0.1 0.3 0.25 0.2 0.25 0.1 0.0 0.15 0.0 0.2 0.25 0.25 0.25 0.05 0.2 0.0 0.0 0.0 0.2 0.0 0.15 0.0

It certainly seems that \lambda is not equal to 0.25 for all plants!
