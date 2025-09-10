[![Open in GitHub Codespaces](
  https://img.shields.io/badge/Open%20in%20GitHub%20Codespaces-333?logo=github)](
  https://codespaces.new/dwave-examples/rna-folding?quickstart=1)
[![Linux/Mac/Windows build status](
  https://circleci.com/gh/dwave-examples/rna-folding.svg?style=shield)](
  https://circleci.com/gh/dwave-examples/rna-folding)

# Improvements to RNA Folding Demo Program.

This is an improvement to the RNA Folding Program based on Fox DM et al "RNA folding using quantum computers".

Two different improvements are implemented. One uses classical reduction techniques to reduce the problem size in various ways, which include mergin maximal stem, removing idle base sequence, and leveraging symmetry. The other includes further factors to the RNA folding process, which is mainly based on the result published on "Turner/Mathews nearest-neighbor (NN)".

# Classical Reduction Techniques

- Maximal Stem
    - Assume that the maximal stem of the bonding matrix must be formed in the optimal secondary structure.
    - Reduce the graph by creating a super-stem consisting of only two bases which are already bonded.
    - Run the problem on the sampler.
    - Retrieve the solution, and convert it back to the solution to the larger problem.
- Idle Base Sequence
    - Repeated bases which is not part of a potential stem can be reduced.
    - Reduce the graph by creating a super-node which does not bond.
    - Run the problem on the sampler.
    - Retrieve the solution, and convert it back to the solution to the larger problem.
- Palindromic symmetry
    - Assume that a long palindromic sequence is likely to fold (form a loop) at its center, creating a hairpin structure.

# Additional Factors to the Hamiltonian

## Variables

- $x_{ij}\in\{0,1\}$: 1 if bases at positions $i<j$ are paired; 0 otherwise.
- Candidates only for pairs with **allowed chemistry** (WC + GU + optional non-canonicals) and **loop length** $⁡j-i-1\ge L_{\min}$ (this is a **pre-filter**, not a constraint).

## Objective (minimize)

Sum of energetic terms (kcal/mol):

- **Nearest-neighbor stacking (bonus, negative):**
    
    $\displaystyle \sum_{(i,j)}\sum_{\text{inner }(i+1,j-1)} \Delta G^{\circ}_{37}\!\big((b_i,b_{i+1});(b_j,b_{j-1})\big)\; x_{ij}x_{i+1,j-1}$
    
- **Isolated-pair penalty (linear, positive):** $\displaystyle \sum_{(i,j)} h_0\, x_{ij}$
- **Hairpin end penalty (applied only at helix ends):**
    
    $\displaystyle \sum_{(i,j)} H(L_{ij})\,x_{ij}\;-\;H(L_{ij})\,x_{ij}x_{i+1,j-1}$
    
- **AU/GU helix-end penalties (per end):**
    
    $\displaystyle \sum_{(i,j)} \text{pen}_{\text{end}}(b_i,b_j)\,x_{ij}\;-\;\text{pen}_{\text{end}}(b_i,b_j)\big[x_{ij}x_{i+1,j-1}+x_{ij}x_{i-1,j+1}\big]$
    
- **Non-canonical pair penalty (linear, positive):** $\displaystyle \sum_{(i,j)\in \text{noncanon}} \lambda_{\text{nc}}\,x_{ij}$
- *(Optional)* **Soft pseudoknot penalty:**
    
    $\displaystyle \sum_{\text{crossings }(i<k<j<l)} \kappa_{\text{soft}}\, x_{ij}x_{kl}$
    

## Hard constraints (CQM)

- **Non-overlap (at most one partner per nucleotide):**
    
    For every position $t$:
    
    $\displaystyle \sum_{(i,j):\,t\in\{i,j\}} x_{ij} \;\le\; 1$
    
- *(Optional)* **No pseudoknots:**
    
    For every crossing pair $(i<k<j<l)$:
    
    $\displaystyle x_{ij} + x_{kl} \;\le\; 1$
    

## Soft constraints (in the objective)

- *(Optional)* **Discourage** (but allow) **pseudoknots:** add $\kappa_{\text{soft}}\,x_{ij}x_{kl}$ for each crossing instead of the hard inequality above.

## Core thermodynamic model

- **Nearest-neighbor (NN) energy model** for RNA helices (ΔG°₃₇ stacks are the dominant stabilizing terms used by modern predictors and databases). Primary source and curated tables: Turner/Mathews **NNDB** (1999 & 2004 RNA sets). [rna.urmc.rochester.edu](https://rna.urmc.rochester.edu/NNDB/?utm_source=chatgpt.com)[Oxford Academic](https://academic.oup.com/nar/article/38/suppl_1/D280/3112245?utm_source=chatgpt.com)
- **Overview of NN model across loop types** (hairpins, internal, bulge, multibranch) and how energy is a sum of stack terms + loop penalties: review by Andronescu et al. (2010). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2995392/?utm_source=chatgpt.com)
- **Current updates to NNDB** and parameter coverage: Mittal et al. (2024) NNDB expansion. [PubMed](https://pubmed.ncbi.nlm.nih.gov/38522645/?utm_source=chatgpt.com)

## Which base pairs are allowed

- **Watson–Crick (AU/UA, GC/CG) and GU wobble** are standard in the Turner 2004 RNA parameterization (with GU-specific stack values). [rna.urmc.rochester.edu](https://rna.urmc.rochester.edu/NNDB/?utm_source=chatgpt.com)
- **GU wobble energetics/structure** (distinct geometry, special contexts): Turner 2004 GU references index; additional structural/thermo studies of GU. [rna.urmc.rochester.edu](https://rna.urmc.rochester.edu/NNDB/turner04/wc-references.html?utm_source=chatgpt.com)[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2553593/?utm_source=chatgpt.com)[Oxford Academic](https://academic.oup.com/nar/article/35/11/3836/2402441?utm_source=chatgpt.com)
- **Non-canonical/mismatches**: internal mismatches and non-WC interactions have measured thermodynamic effects; allowing them (with penalties) is consistent with Turner-era datasets (e.g., mismatches next to GC) and reviews. [rna.urmc.rochester.edu](https://rna.urmc.rochester.edu/NNDB/turner04/wc-references.html?utm_source=chatgpt.com)[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2995392/?utm_source=chatgpt.com)

## Stacking (quadratic bonuses in the QUBO)

- **Stacking free energies (ΔG°₃₇) for adjacent pairs** are the backbone of helix stability in the Turner 2004 set. Predictors like RNAstructure/ViennaRNA rely on these tables. [rna.urmc.rochester.edu](https://rna.urmc.rochester.edu/NNDB/?utm_source=chatgpt.com)[BioMed Central](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-129?utm_source=chatgpt.com)

## Hairpin loop penalties (applied at helix ends)

- **Hairpin initiation penalties depend on loop length L** and the **closing base pair**; classic experiments (Serra & Turner) quantify these trends. We encode them at helix ends via a quadratic “cancellation” trick in the QUBO. [PubMed+1](https://pubmed.ncbi.nlm.nih.gov/7690127/?utm_source=chatgpt.com)
- **Sequence-specific bonuses** for stable tetraloops (e.g., GNRA, UNCG, CUUG) are well-documented; we left motif bonuses optional but the literature supports adding them later. [PMC+1](https://pmc.ncbi.nlm.nih.gov/articles/PMC2811670/?utm_source=chatgpt.com)[Oxford Academic](https://academic.oup.com/nar/article/27/5/1398/2902354?utm_source=chatgpt.com)

## AU/GU helix-end penalties (per end)

- Turner models include **end corrections** (AU/GU ends less stable than GC ends). Our QUBO captures this with per-end penalties that are canceled for interior pairs (i.e., only termini pay). See NNDB descriptions and Mathews/Turner reviews. [rna.urmc.rochester.edu](https://rna.urmc.rochester.edu/NNDB/?utm_source=chatgpt.com)[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4086783/?utm_source=chatgpt.com)[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0959440X06000819?utm_source=chatgpt.com)

## Hard constraints

- **Non-overlap (each nucleotide pairs ≤1)** is a fundamental combinatorial constraint in RNA 2D models; standard in DP and optimization formulations. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0959440X06000819?utm_source=chatgpt.com)
- **Anti-pseudoknot (optional)**: forbidding crossings yields the classic nested secondary-structure class; allowing pseudoknots makes the problem NP-complete for broad model families (hence we keep it optional with a penalty). [PubMed](https://pubmed.ncbi.nlm.nih.gov/11108471/?utm_source=chatgpt.com)

# How to run

1. Setup DWave-OCEAN API configuration
2. Download requirements
```code
pip install -r requirements.txt
```
3. Go to ```RNA_folding.py```, and edit the main method. You can run sequence by sequence, or by files. 
- When running sequence by sequence, change the ```PBD_ID```. 
- When running multiple sequence in file, change the filename in ```process_sequence_list("filename.txt")```.

4. Run the code
```python RNA_folding.py```

5. Check the results

The plot is saved in the plots/ directory.
The result is saved in the results/ directory.


### Bibliography

- **Mathews, Sabina, Zuker & Turner (1999)** — *Expanded sequence dependence of thermodynamic parameters improves prediction of RNA secondary structure.* **J. Mol. Biol.** 288:911–940.
    
    Foundational NN free-energy set for RNA (ΔG°₃₇), expanding sequence dependence and improving folding predictions. [PubMed](https://pubmed.ncbi.nlm.nih.gov/10329189/?utm_source=chatgpt.com)
    
- **Turner & Mathews (2010)** — *NNDB: the nearest neighbor parameter database for predicting stability of nucleic acid secondary structure.* **Nucleic Acids Research** 38:D280–D282.
    
    Official database paper documenting the 1999 & 2004 RNA NN parameter sets and usage. (Also see the live NNDB site.) [PubMed](https://pubmed.ncbi.nlm.nih.gov/19880381/?utm_source=chatgpt.com)[Mathews Lab](https://rna.urmc.rochester.edu/NNDB/?utm_source=chatgpt.com)
    
- **Turner (NNDB 2004 collection)** — *Turner 2004 RNA folding parameters (web compendium: helices, GU pairs, dangling ends, hairpins, internal/bulge/multibranch loops, coaxial stacking).*
    
    Curated parameter pages that aggregate the 2004 RNA NN rules used widely in practice. [Mathews Lab](https://rna.urmc.rochester.edu/NNDB/turner04/index.html?utm_source=chatgpt.com)
    
- **Serra & Turner (1993)** — *RNA hairpin loop stability depends on closing base pair.* **Nucleic Acids Research** 21:3845–3849.
    
    Classic experiments establishing loop (hairpin) penalties and dependence on the closing pair—motivates our hairpin end penalty. [Oxford Academic](https://academic.oup.com/nar/article/21/16/3845/2386373?utm_source=chatgpt.com)[PubMed](https://pubmed.ncbi.nlm.nih.gov/7690127/?utm_source=chatgpt.com)
    
- **Lu, Turner & Mathews (2006)** — *A set of nearest neighbor parameters for predicting the enthalpy change of RNA secondary structure formation.* **Nucleic Acids Research** 34:4912–4924.
    
    Complements free-energy tables with ΔH°; useful for temperature scaling beyond 37 °C. [Oxford Academic](https://academic.oup.com/nar/article/34/17/4912/3111941?utm_source=chatgpt.com)
    
- **Spasic, Warner, Jonikas & Mathews (2018)** — *Improving RNA nearest neighbor parameters for helices by leveraging Inosine substitutions.* **Nucleic Acids Research** 46:4883–4892.
    
    Representative of modern refinements to NN helix parameters; shows the lineage of updates to Turner-style models. [Oxford Academic](https://academic.oup.com/nar/article/46/10/4883/4990632?utm_source=chatgpt.com)