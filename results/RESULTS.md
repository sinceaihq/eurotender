# EuroTender evaluation results

## Validation split

### Model, CPV division (n = 79,714, label coverage 1.000)

| subset | top-1 | top-3 | top-5 | macro-F1 | weighted-F1 | macro-P | macro-R | ECE |
|---|---|---|---|---|---|---|---|---|
| all | 0.828 | 0.923 | 0.944 | 0.684 | 0.823 | 0.753 | 0.648 | 0.048 |
| novel only (87.9%) | 0.810 | 0.914 | 0.938 | 0.657 | 0.804 | 0.724 | 0.622 | 0.052 |

### Model, CPV group (n = 79,714, label coverage 1.000)

| subset | top-1 | top-3 | top-5 | macro-F1 | weighted-F1 | macro-P | macro-R | ECE |
|---|---|---|---|---|---|---|---|---|
| all | 0.666 | 0.815 | 0.851 | 0.460 | 0.653 | 0.561 | 0.422 | 0.116 |
| novel only (87.9%) | 0.635 | 0.795 | 0.836 | 0.413 | 0.619 | 0.502 | 0.382 | 0.124 |

### Baselines vs. model (CPV division)

| system | top-1 | top-5 | macro-F1 | weighted-F1 |
|---|---|---|---|---|
| majority class | 0.166 | 0.495 | 0.006 | 0.047 |
| CPV-label similarity (zero-shot) | 0.346 | 0.703 | 0.235 | 0.373 |
| Complement NB (same features) | 0.632 | 0.895 | 0.405 | 0.607 |
| **EuroTender (this model)** | **0.828** | **0.944** | **0.684** | **0.823** |

## Test split

### Model, CPV division (n = 234,471, label coverage 1.000)

| subset | top-1 | top-3 | top-5 | macro-F1 | weighted-F1 | macro-P | macro-R | ECE |
|---|---|---|---|---|---|---|---|---|
| all | 0.821 | 0.920 | 0.942 | 0.671 | 0.815 | 0.739 | 0.636 | 0.049 |
| novel only (90.3%) | 0.809 | 0.914 | 0.938 | 0.656 | 0.803 | 0.724 | 0.621 | 0.051 |

### Model, CPV group (n = 234,471, label coverage 1.000)

| subset | top-1 | top-3 | top-5 | macro-F1 | weighted-F1 | macro-P | macro-R | ECE |
|---|---|---|---|---|---|---|---|---|
| all | 0.650 | 0.803 | 0.841 | 0.441 | 0.635 | 0.555 | 0.401 | 0.122 |
| novel only (90.3%) | 0.629 | 0.791 | 0.830 | 0.411 | 0.613 | 0.521 | 0.374 | 0.126 |

### Baselines vs. model (CPV division)

| system | top-1 | top-5 | macro-F1 | weighted-F1 |
|---|---|---|---|---|
| majority class | 0.174 | 0.500 | 0.007 | 0.052 |
| CPV-label similarity (zero-shot) | 0.341 | 0.698 | 0.230 | 0.367 |
| Complement NB (same features) | 0.615 | 0.886 | 0.394 | 0.589 |
| **EuroTender (this model)** | **0.821** | **0.942** | **0.671** | **0.815** |

## Per-language results (test, CPV division)

| language | n | top-1 | top-3 | macro-F1 |
|---|---|---|---|---|
| deu | 58,254 | 0.876 | 0.951 | 0.655 |
| fra | 39,932 | 0.798 | 0.909 | 0.615 |
| pol | 27,618 | 0.879 | 0.953 | 0.747 |
| spa | 16,340 | 0.790 | 0.907 | 0.650 |
| eng | 12,020 | 0.702 | 0.849 | 0.580 |
| ita | 11,194 | 0.809 | 0.914 | 0.631 |
| nld | 9,035 | 0.753 | 0.885 | 0.594 |
| ces | 8,059 | 0.833 | 0.932 | 0.690 |
| swe | 7,850 | 0.719 | 0.860 | 0.551 |
| ron | 6,964 | 0.824 | 0.928 | 0.641 |
| por | 5,466 | 0.794 | 0.900 | 0.622 |
| bul | 5,069 | 0.872 | 0.949 | 0.658 |
| lit | 4,111 | 0.805 | 0.902 | 0.638 |
| fin | 3,688 | 0.772 | 0.898 | 0.615 |
| ell | 3,464 | 0.768 | 0.881 | 0.591 |
| lav | 2,920 | 0.813 | 0.910 | 0.619 |
| slv | 2,533 | 0.799 | 0.903 | 0.647 |
| hrv | 2,491 | 0.813 | 0.919 | 0.653 |
| hun | 2,187 | 0.803 | 0.899 | 0.620 |
| dan | 1,980 | 0.782 | 0.883 | 0.529 |
| slk | 1,831 | 0.802 | 0.895 | 0.618 |
| est | 1,453 | 0.791 | 0.882 | 0.576 |

## Per-division results (test)

| CPV | label | support | precision | recall | F1 |
|---|---|---|---|---|---|
| 45 | Construction work | 40,813 | 0.907 | 0.949 | 0.928 |
| 71 | Architectural, construction, engineering and inspection services | 24,842 | 0.865 | 0.888 | 0.876 |
| 33 | Medical equipments, pharmaceuticals and personal care products | 23,350 | 0.884 | 0.954 | 0.917 |
| 90 | Sewage, refuse, cleaning and environmental services | 16,484 | 0.898 | 0.913 | 0.905 |
| 34 | Transport equipment and auxiliary products to transportation | 11,809 | 0.807 | 0.853 | 0.829 |
| 79 | Business services: law, marketing, consulting, recruitment, printing and security | 11,558 | 0.726 | 0.751 | 0.738 |
| 50 | Repair and maintenance services | 11,263 | 0.772 | 0.827 | 0.799 |
| 72 | IT services: consulting, software development, Internet and support | 10,085 | 0.680 | 0.746 | 0.712 |
| 39 | Furniture (incl. office furniture), furnishings, domestic appliances (excl. lighting) and cleaning products | 5,839 | 0.774 | 0.717 | 0.744 |
| 48 | Software package and information systems | 5,574 | 0.658 | 0.593 | 0.624 |
| 38 | Laboratory, optical and precision equipments (excl. glasses) | 5,513 | 0.720 | 0.736 | 0.728 |
| 66 | Financial and insurance services | 5,481 | 0.929 | 0.935 | 0.932 |
| 30 | Office and computing machinery, equipment and supplies except furniture and software packages | 5,325 | 0.742 | 0.720 | 0.731 |
| 60 | Transport services (excl. Waste transport) | 4,666 | 0.856 | 0.887 | 0.871 |
| 09 | Petroleum products, fuel, electricity and other sources of energy | 4,663 | 0.891 | 0.936 | 0.913 |
| 77 | Agricultural, forestry, horticultural, aquacultural and apicultural services | 4,437 | 0.864 | 0.863 | 0.864 |
| 44 | Construction structures and materials; auxiliary products to construction (except electric apparatus) | 3,540 | 0.682 | 0.569 | 0.621 |
| 15 | Food, beverages, tobacco and related products | 3,463 | 0.872 | 0.883 | 0.877 |
| 42 | Industrial machinery | 3,245 | 0.602 | 0.448 | 0.513 |
| 31 | Electrical machinery, apparatus, equipment and consumables; lighting | 3,146 | 0.674 | 0.602 | 0.636 |
| 85 | Health and social work services | 2,856 | 0.789 | 0.717 | 0.751 |
| 32 | Radio, television, communication, telecommunication and related equipment | 2,615 | 0.609 | 0.550 | 0.578 |
| 55 | Hotel, restaurant and retail trade services | 2,453 | 0.810 | 0.848 | 0.829 |
| 98 | Other community, social and personal services | 2,159 | 0.753 | 0.604 | 0.670 |
| 35 | Security, fire-fighting, police and defence equipment | 1,917 | 0.630 | 0.453 | 0.527 |
| 18 | Clothing, footwear, luggage articles and accessories | 1,665 | 0.730 | 0.778 | 0.753 |
| 80 | Education and training services | 1,636 | 0.734 | 0.744 | 0.739 |
| 92 | Recreational, cultural and sporting services | 1,474 | 0.627 | 0.556 | 0.590 |
| 24 | Chemical products | 1,431 | 0.833 | 0.723 | 0.774 |
| 63 | Supporting and auxiliary transport services; travel agencies services | 1,426 | 0.736 | 0.543 | 0.625 |
| 64 | Postal and telecommunications services | 1,323 | 0.778 | 0.690 | 0.731 |
| 73 | Research and development services and related consultancy services | 1,101 | 0.565 | 0.448 | 0.500 |
| 75 | Administration, defence and social security services | 982 | 0.584 | 0.196 | 0.293 |
| 43 | Machinery for mining, quarrying, construction equipment | 869 | 0.658 | 0.482 | 0.556 |
| 22 | Printed matter and related products | 824 | 0.738 | 0.593 | 0.658 |
| 03 | Agricultural, farming, fishing, forestry and related products | 777 | 0.760 | 0.546 | 0.635 |
| 16 | Agricultural machinery | 692 | 0.774 | 0.634 | 0.697 |
| 51 | Installation services (except software) | 682 | 0.495 | 0.136 | 0.214 |
| 37 | Musical instruments, sport goods, games, toys, handicraft, art materials and accessories | 649 | 0.773 | 0.521 | 0.622 |
| 70 | Real estate services | 583 | 0.709 | 0.460 | 0.558 |
| 65 | Public utilities | 437 | 0.639 | 0.336 | 0.441 |
| 14 | Mining, basic metals and related products | 428 | 0.858 | 0.577 | 0.690 |
| 19 | Leather and textile fabrics, plastic and rubber materials | 262 | 0.747 | 0.416 | 0.534 |
| 76 | Services related to the oil and gas industry | 94 | 0.645 | 0.213 | 0.320 |
| 41 | Collected and purified water | 40 | 0.500 | 0.100 | 0.167 |

## Most frequent confusions (test, CPV division)

| true | predicted | count | share of true class |
|---|---|---|---|
| 48 Software package and information systems | 72 IT services: consulting, software development, Internet and support | 1,322 | 23.7% |
| 71 Architectural, construction, engineering and inspection services | 45 Construction work | 932 | 3.8% |
| 72 IT services: consulting, software development, Internet and support | 48 Software package and information systems | 839 | 8.3% |
| 79 Business services: law, marketing, consulting, recruitment, printing and security | 71 Architectural, construction, engineering and inspection services | 763 | 6.6% |
| 45 Construction work | 71 Architectural, construction, engineering and inspection services | 694 | 1.7% |
| 38 Laboratory, optical and precision equipments (excl. glasses) | 33 Medical equipments, pharmaceuticals and personal care products | 675 | 12.2% |
| 79 Business services: law, marketing, consulting, recruitment, printing and security | 72 IT services: consulting, software development, Internet and support | 530 | 4.6% |
| 72 IT services: consulting, software development, Internet and support | 79 Business services: law, marketing, consulting, recruitment, printing and security | 478 | 4.7% |
| 90 Sewage, refuse, cleaning and environmental services | 71 Architectural, construction, engineering and inspection services | 432 | 2.6% |
| 71 Architectural, construction, engineering and inspection services | 79 Business services: law, marketing, consulting, recruitment, printing and security | 410 | 1.7% |
| 45 Construction work | 50 Repair and maintenance services | 396 | 1.0% |
| 71 Architectural, construction, engineering and inspection services | 50 Repair and maintenance services | 378 | 1.5% |
| 30 Office and computing machinery, equipment and supplies except furniture and software packages | 48 Software package and information systems | 351 | 6.6% |
| 50 Repair and maintenance services | 45 Construction work | 309 | 2.7% |
| 48 Software package and information systems | 30 Office and computing machinery, equipment and supplies except furniture and software packages | 307 | 5.5% |

## Confidence vs. accuracy (test, CPV division)

| min. confidence | coverage | top-1 |
|---|---|---|
| 0.0 | 1.000 | 0.821 |
| 0.5 | 0.834 | 0.899 |
| 0.7 | 0.753 | 0.926 |
| 0.9 | 0.633 | 0.954 |

## Accuracy vs. similarity to the nearest training notice (test sample)

| nearest-train cosine | n | top-1 | top-5 |
|---|---|---|---|
| 0.0–0.5 | 1,425 | 0.741 | 0.909 |
| 0.5–0.7 | 885 | 0.868 | 0.956 |
| 0.7–0.8 | 274 | 0.887 | 0.949 |
| 0.8–0.9 | 190 | 0.937 | 0.968 |
| 0.9–1.0 | 226 | 0.965 | 0.987 |

## Inference performance (CPU)

- model load: 0.05 s
- single-text latency: p50 1.7 ms, p95 6.2 ms
- batch throughput: 661 notices/s (batch of 5,000, one process)
- peak RSS of a fresh process (heads loaded: division; 5k texts): 748 MB
- files: config.json 0.0 MB, model.safetensors 109.8 MB, group.safetensors 274.7 MB
