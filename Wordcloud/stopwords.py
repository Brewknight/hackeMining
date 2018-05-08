STOPWORDS = """
a about above across after afterwards again against all almost alone along already also although always am among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere are around as at back be
became because become becomes becoming been before beforehand behind being below beside besides between beyond bill both bottom but by call can
cannot cant co computer con could couldnt cry de describe
detail did didn do does doesn doing don done down due during
each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything everywhere except few fifteen
fify fill find fire first five for former formerly forty found four from front full further get give go got
had has hasnt have he hence her here hereafter hereby herein hereupon hers herself him himself his how however hundred i ie
if I i'm I'm in inc indeed interest into is it its itself keep last latter latterly least less like ltd
just
kg km
made make many may me meanwhile might mill mine more moreover most mostly move much must my myself name namely
n't neither never nevertheless next nine no nobody none noone nor not nothing now nowhere of off
often on once one only onto or other others otherwise our ours ourselves out over own part per
perhaps please put rather re
quite
rather really regarding
same said say saying says see seem seemed seeming seems serious several she should show side since sincere six sixty so some somehow someone something sometime sometimes somewhere still such system take ten
than that the their them themselves then thence there thereafter thereby therefore therein thereupon these they thick thin thing third this those though three through throughout thru thus to together too top toward towards twelve twenty two un under
until up unless upon us used using
various very very via
was we well were what whatever when whence whenever where whereafter whereas whereby wherein whereupon wherever whether which while whither who whoever whole whom whose why will with within without would yet year you
your yours yourself yourselves
. , < > / ? [ ] { } + = _ - ) ( | \\ \t \n " ' ; : ` ~ ! @ # $ % ^ &
"""
STOPWORDS = set(w for w in STOPWORDS.split() if w)