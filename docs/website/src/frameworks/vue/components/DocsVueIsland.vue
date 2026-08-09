<script setup lang="ts">
defineProps<{
  title?: string;
  summary?: string;
}>();
</script>

<template>
  <section class="docs-vue-island" aria-label="Vue documentation island">
    <p class="kicker">Framework island · Vue</p>
    <h2>{{ title ?? 'Vue island' }}</h2>
    <p class="lede">
      {{
        summary ??
        'Standalone Vue surface for the docs dashboard multi-framework experiments (mount via frameworks/vue/mount.ts).'
      }}
    </p>
  </section>
</template>

<style scoped>
.docs-vue-island {
  padding: 1rem 1.1rem;
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 10px;
  background: rgba(15, 23, 42, 0.55);
  color: #e2e8f0;
  font-family: system-ui, sans-serif;
}
.kicker {
  margin: 0 0 0.25rem;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #94a3b8;
}
h2 {
  margin: 0;
  font-size: 1.1rem;
}
.lede {
  margin: 0.4rem 0 0;
  color: #94a3b8;
  font-size: 0.9rem;
}
</style>
