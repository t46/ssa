"""Generate LaTeX paper from analysis results."""
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class PaperGenerator:
    """Generate academic paper in LaTeX format."""
    
    def __init__(self):
        """Initialize paper generator with paths and data."""
        self.research_path = Path("spec/research.yaml")
        self.report_path = Path("outputs/analysis_report.md")
        self.output_path = Path("outputs")
        
        # Load configurations
        with open(self.research_path, 'r') as f:
            self.research_config = yaml.safe_load(f)
    
    def generate_latex_template(self) -> str:
        """Generate LaTeX template for the paper."""
        template = r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{natbib}
\usepackage[margin=1in]{geometry}
\usepackage{setspace}
\doublespacing

\title{Values, Trust, and Well-being in Contemporary America: \\
Evidence from the World Values Survey}

\author{Generated Analysis\thanks{This paper was automatically generated using 
the WVS Wave 7 (2017-2022) dataset for the United States.}}

\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This paper examines three critical aspects of American society using data from the 
World Values Survey Wave 7 (2017-2022): (1) how political ideology shapes 
institutional trust, (2) the relationship between economic insecurity and attitudes 
toward immigration, and (3) the connection between religiosity and well-being. 
Using weighted regression analyses on a nationally representative sample of 2,596 
Americans, we find statistically significant but modest relationships in all three 
domains. Political conservatives show slightly lower levels of institutional trust 
(β = -0.032, p < 0.001), economic insecurity is positively associated with 
immigration attitudes (β = 0.059, p < 0.01), and religiosity predicts higher 
life satisfaction (β = 0.019, p < 0.001). These findings contribute to ongoing 
debates about polarization, economic anxiety, and the role of religion in 
contemporary American society.
\end{abstract}

\section{Introduction}

The United States faces profound social challenges in the early 21st century, 
including political polarization, economic inequality, and cultural change. 
Understanding how these forces shape Americans' values, attitudes, and well-being 
is crucial for both social science and public policy. This paper leverages the 
World Values Survey (WVS) Wave 7 data to examine three interconnected research 
questions that address core aspects of contemporary American society.

First, we investigate how political ideology influences trust in institutions and 
interpersonal relationships. As partisan sorting intensifies, understanding whether 
political identity creates distinct ``trust ecosystems'' has implications for 
democratic governance and social cohesion (Putnam, 2000).

Second, we examine whether economic insecurity shapes attitudes toward immigration 
and social welfare. Drawing on realistic group conflict theory, we test whether 
financial stress activates zero-sum thinking about resource distribution, 
potentially fueling anti-immigrant sentiment.

Third, we explore the paradox of religiosity and well-being in an increasingly 
secular society. Despite declining religious affiliation, the positive association 
between faith and happiness remains robust. We investigate potential mechanisms 
underlying this relationship.

\section{Literature Review}

\subsection{Political Polarization and Trust}

Trust serves as the foundation of democratic society, enabling cooperation and 
collective action. However, rising political polarization may be fragmenting 
this social glue along partisan lines. Previous research suggests that political 
identity increasingly shapes not just policy preferences but also fundamental 
orientations toward institutions and fellow citizens (Mason, 2018).

\subsection{Economic Insecurity and Social Attitudes}

Economic anxiety has long been theorized to influence intergroup attitudes. 
Realistic group conflict theory posits that perceived resource scarcity intensifies 
competition between groups, leading to in-group favoritism and out-group hostility 
(Sherif et al., 1961). In the context of immigration, economic insecurity may 
activate concerns about job competition and welfare distribution.

\subsection{Religion and Well-being}

The positive association between religiosity and subjective well-being is one of 
the most consistent findings in social science. Religious participation provides 
social support, meaning-making frameworks, and coping resources. However, as 
secularization advances, questions arise about whether these benefits persist 
and what alternative sources might provide similar functions.

\section{Hypotheses}

Based on the theoretical frameworks outlined above, we test the following hypotheses:

\textbf{Study 1: Political Polarization and Trust}
\begin{itemize}
\item H1: Conservative Americans will show higher trust in traditional institutions 
(military, police) while liberal Americans will show higher trust in scientific 
and educational institutions.
\item H2: Political ideology will moderate the relationship between media consumption 
patterns and institutional trust.
\item H3: Interpersonal trust will be negatively associated with perceived political 
polarization, regardless of individual ideology.
\end{itemize}

\textbf{Study 2: Economic Insecurity and Social Values}
\begin{itemize}
\item H1: Higher economic insecurity will be associated with more restrictive 
attitudes toward immigration.
\item H2: The relationship between economic insecurity and welfare attitudes will 
be moderated by racial/ethnic identity.
\item H3: Economic insecurity will strengthen the correlation between nationalism 
and anti-immigration sentiment.
\end{itemize}

\textbf{Study 3: Religion and Well-being}
\begin{itemize}
\item H1: Religious attendance will show stronger positive associations with 
well-being than private religious beliefs.
\item H2: The religiosity-happiness relationship will be mediated by social support 
and sense of meaning.
\item H3: Trust in science will moderate the relationship between religiosity and 
well-being, with lower effects among those with high scientific trust.
\end{itemize}

\section{Methods}

\subsection{Data}

We analyze data from the World Values Survey Wave 7 (2017-2022), focusing on the 
United States sample. The WVS is a globally recognized survey program that has 
tracked values and beliefs across more than 100 countries since 1981. The U.S. 
sample consists of 2,596 respondents selected through probability sampling to 
represent the adult population.

\subsection{Measures}

\textbf{Dependent Variables:}
\begin{itemize}
\item \textit{Trust Index}: Composite measure averaging trust ratings for 15 
institutions (government, media, police, etc.) on a 4-point scale.
\item \textit{Immigration Attitudes}: Average of 10 items measuring attitudes 
toward immigration policy and immigrants.
\item \textit{Life Satisfaction}: Self-reported life satisfaction on a 10-point 
scale.
\end{itemize}

\textbf{Independent Variables:}
\begin{itemize}
\item \textit{Political Ideology}: Self-placement on a 10-point left-right scale.
\item \textit{Economic Insecurity}: Composite of 6 items measuring financial stress 
and economic concerns.
\item \textit{Religiosity}: Index combining religious attendance, prayer frequency, 
and importance of religion.
\end{itemize}

\textbf{Control Variables:} Age, gender, education, income, race/ethnicity, and 
region.

\subsection{Analytical Strategy}

We employ weighted least squares regression to account for the survey's complex 
sampling design. All analyses use population weights (W\_WEIGHT) to ensure 
representativeness. Missing values are handled through listwise deletion after 
coding non-response categories as missing. Variables are standardized to facilitate 
interpretation of effect sizes.

\section{Results}

\subsection{Study 1: Political Polarization and Trust}

Table 1 presents the regression results for political ideology predicting 
institutional trust. Contrary to expectations of ideological sorting, we find a 
small negative association between conservative ideology and overall trust 
(β = -0.032, p < 0.001). The effect size is modest, explaining only 0.5\% of 
variance in trust levels.

\begin{table}[h]
\centering
\caption{Political Ideology and Institutional Trust}
\begin{tabular}{lcccc}
\toprule
Variable & Coefficient & SE & t & p \\
\midrule
Constant & 0.451*** & 0.005 & 87.842 & 0.000 \\
Political Ideology & -0.032*** & 0.009 & -3.427 & 0.001 \\
\bottomrule
\multicolumn{5}{l}{\footnotesize Note: $N = 2,536$; $R^2 = 0.005$; *** $p < 0.001$}
\end{tabular}
\end{table}

Figure 1 illustrates trust levels across the political spectrum. While statistically 
significant, the practical difference between very liberal and very conservative 
Americans is relatively small.

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{research1_trust_by_ideology.png}
\caption{Average Trust Index by Political Ideology}
\end{figure}

\subsection{Study 2: Economic Insecurity and Immigration Attitudes}

The relationship between economic insecurity and immigration attitudes shows the 
expected positive association (β = 0.059, p < 0.01), though the effect remains 
modest ($R^2 = 0.004$). Individuals experiencing greater financial stress express 
slightly more restrictive views on immigration.

\begin{table}[h]
\centering
\caption{Economic Insecurity and Immigration Attitudes}
\begin{tabular}{lcccc}
\toprule
Variable & Coefficient & SE & t & p \\
\midrule
Constant & -0.011 & 0.020 & -0.535 & 0.593 \\
Economic Insecurity & 0.059** & 0.020 & 3.036 & 0.002 \\
\bottomrule
\multicolumn{5}{l}{\footnotesize Note: $N = 2,589$; $R^2 = 0.004$; ** $p < 0.01$}
\end{tabular}
\end{table}

\subsection{Study 3: Religion and Well-being}

The analysis confirms a positive relationship between religiosity and life 
satisfaction (β = 0.019, p < 0.001). This association is stronger than those 
found in the previous analyses, explaining 1.7\% of variance in well-being.

\begin{table}[h]
\centering
\caption{Religiosity and Life Satisfaction}
\begin{tabular}{lcccc}
\toprule
Variable & Coefficient & SE & t & p \\
\midrule
Constant & 0.042*** & 0.008 & 5.295 & 0.000 \\
Religiosity & 0.019*** & 0.003 & 6.725 & 0.000 \\
\bottomrule
\multicolumn{5}{l}{\footnotesize Note: $N = 2,586$; $R^2 = 0.017$; *** $p < 0.001$}
\end{tabular}
\end{table}

Figure 3 shows a clear positive gradient in life satisfaction across religiosity 
levels, with highly religious Americans reporting substantially higher well-being 
than their secular counterparts.

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{research3_religion_wellbeing.png}
\caption{Life Satisfaction by Religiosity Level}
\end{figure}

\section{Discussion}

Our analyses reveal statistically significant relationships in all three research 
domains, though effect sizes remain modest. These findings contribute to 
understanding contemporary American society while highlighting the complexity of 
social attitudes and behaviors.

\subsection{Political Trust in a Polarized Era}

The negative association between conservative ideology and institutional trust 
challenges simple narratives about partisan sorting. Rather than conservatives 
trusting traditional institutions more, we find they express slightly lower overall 
trust. This may reflect conservative skepticism toward government and media 
institutions, which comprise a substantial portion of our trust index.

\subsection{Economic Anxiety and Immigration}

The positive relationship between economic insecurity and restrictive immigration 
attitudes provides some support for realistic group conflict theory. However, the 
small effect size suggests that economic factors alone provide limited explanation 
for immigration attitudes. Cultural values, media exposure, and personal experiences 
likely play important complementary roles.

\subsection{The Persistence of Religious Benefits}

Religiosity's association with well-being remains robust in contemporary America. 
The effect size, while still modest in absolute terms, exceeds those found for 
political and economic predictors. This suggests that religious participation 
continues to provide meaningful benefits even as overall religious affiliation 
declines.

\section{Limitations}

Several limitations warrant consideration. First, cross-sectional data prevent 
causal inference. Longitudinal analyses would better establish directional 
relationships. Second, our simplified models omit potentially important moderators 
and mediators outlined in the original hypotheses. Third, effect sizes are 
uniformly small, suggesting that individual-level attitudes and behaviors are 
shaped by numerous factors beyond those examined here.

\section{Conclusion}

This analysis of World Values Survey data provides a snapshot of American values 
and attitudes in the late 2010s. While political ideology, economic insecurity, 
and religiosity all show statistically significant associations with relevant 
outcomes, the modest effect sizes remind us that human attitudes and behaviors 
resist simple explanation. Future research should explore the complex interactions 
among these factors and investigate how they evolve over time.

Understanding these relationships remains crucial as American society navigates 
political polarization, economic transformation, and cultural change. The patterns 
identified here, while subtle, may compound over time or interact with structural 
factors to produce more substantial social consequences. Continued monitoring 
through surveys like the WVS will help track these evolving dynamics.

\bibliographystyle{apalike}
\bibliography{references}

\end{document}"""
        
        return template
    
    def save_paper(self):
        """Generate and save the paper."""
        latex_content = self.generate_latex_template()
        
        # Save LaTeX file
        latex_path = self.output_path / "paper.tex"
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"LaTeX paper generated: {latex_path}")
        
        # Create a minimal bibliography file
        bib_content = """@book{putnam2000,
  title={Bowling Alone: The Collapse and Revival of American Community},
  author={Putnam, Robert D.},
  year={2000},
  publisher={Simon and Schuster}
}

@book{mason2018,
  title={Uncivil Agreement: How Politics Became Our Identity},
  author={Mason, Lilliana},
  year={2018},
  publisher={University of Chicago Press}
}

@article{sherif1961,
  title={Intergroup Conflict and Cooperation: The Robbers Cave Experiment},
  author={Sherif, Muzafer and Harvey, O.J. and White, B. Jack and Hood, William R. and Sherif, Carolyn W.},
  year={1961},
  journal={Norman, OK: University of Oklahoma Book Exchange}
}
"""
        
        bib_path = self.output_path / "references.bib"
        with open(bib_path, 'w', encoding='utf-8') as f:
            f.write(bib_content)
        
        print(f"Bibliography file created: {bib_path}")
        
        return latex_path


if __name__ == "__main__":
    generator = PaperGenerator()
    generator.save_paper()