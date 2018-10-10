# All I need for data preparation here

## linearize.py

It builds up a lookup table indexed by (observation_id, sentence_id)

Basically it allows you to find the linearized representation and the duration of a sentence.

The name "linearize.py" might be confusing, but really that's the bulk of work done here.

## materialize.py

I hope you've all got used to my taste of naming.

This scripts basically pairs audio wav files with linearized syntax.

I call it "materialize.py" because we can turn speech timing into actual, material audio data.