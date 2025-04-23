# HARP HOA Dataset
HARP: A Large-Scale Higher-Order Ambisonic Room Impulse Response Dataset

Accepted at ICASSP GenDA 2025 workshop.

Provided code for generating HOA RIRs using ISM. 
This is first version of HARP that uses SphericalHarmonicDirectivity class added to Pyroomacoustics library to generate 7th order HOA.
The details can be found in the paper.

If requirements are met, just run Generate.py to generate RIRs. Setting parameters num_rooms and num_positions allows you to change the number of rooms and positions in each room to be generated.

Since the data generated for 100,000 RIRs is more than 1TB, it might take some time to be uploaded.

Version 2 is uploaded that does not require SphericalHarmonicDirectivity and has following pros and cons:
+ Ray Tracing
+ Complex Geometries
+ Eigenmike style microphone configuration
- No higher bound for Reverberation time (can be added).

Cite this work: 

@misc{saini2025harp,
      title={HARP: A Large-Scale Higher-Order Ambisonic Room Impulse Response Dataset}, 
      author={Shivam Saini and Jürgen Peissig},
      year={2025},
      eprint={2411.14207},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2411.14207}, 
}
