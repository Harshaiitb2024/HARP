# HARP HOA Dataset
HARP: A Large-Scale Higher-Order Ambisonic Room Impulse Response Dataset

Accepted at ICASSP GenDA 2025 workshop.

Provided code for generating HOA RIRs using ISM. 
This is first version of HARP that uses SphericalHarmonicDirectivity class added to Pyroomacoustics library to generate 7th order HOA.
The details can be found in the paper.

If requirements are met, just run Generate.py to generate RIRs as follow:

      python Generate.py --output_path "./Generated_HOA_IRs/" --num_rooms 10000 --num_positions 10 --ambi_order 7 --sample_rate 48000

Since the data generated for 100,000 RIRs is more than 1TB, it might take some time to be uploaded.

In Addition, V2 is provided that uses Ray Tracing in Complex Geometries using a spherical array, raw RIRs can be saved for now, the same command can be used inside v2 directory to generate RIRs.

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
