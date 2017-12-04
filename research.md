---
layout: page
title: "Research"
description: "Summary of some of my research projects"
header-img: "img/research-bg.jpg"
order: 2
comments: false
---

Unfortunately still under construction :sleepy:. I'm currently focusing on the [Machine Learning Glossary](/machine-learning-glossary/){:.mdLink}.

In the meantime check out my <a class="mdLink" href="https://github.com/{{ site.github_username }}"> github account</a> and if you want more information concering some of my work don't hesitate to shoot me a <a class="mdLink" href="mailto:{{ site.email_username }}"> mail</a> !
  

{% for post in site.categories.research %}
<div class="post-preview">
    <a href="{{ post.url | prepend: site.baseurl }}">
        <h2 class="post-title">            {{ post.title }}
        </h2>
        {% if post.subtitle %}
        <h3 class="post-subtitle">
            {{ post.subtitle }}
        </h3>
        {% endif %}
    </a>
    <p class="post-meta">{{ post.date | date: "%B %-d, %Y" }}</p>
    <p class="description">{% if post.description %}{{ post.description | strip_html | strip_newlines | truncate: 250 }}{% else %}{{ post.content | strip_html | strip_newlines | truncate: 250 }}{% endif %}</p>
</div>
<hr>
{% endfor %}