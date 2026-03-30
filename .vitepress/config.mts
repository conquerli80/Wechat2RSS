import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Wechat2RSS",
  description: "微信公众号RSS服务",
  lang: 'zh-CN',
  sitemap: {
    hostname: 'https://wechat2rss.xlab.app'
  },
  lastUpdated: true,
  head: [
    [
      'script',
      { async: '', src: 'https://www.googletagmanager.com/gtag/js?id=G-H7VDKD5SEH' }
    ],
    [
      'script',
      {},
      `window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-H7VDKD5SEH');`
    ]
  ],
  themeConfig: {
    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换',
              closeText: '关闭'
            }
          }
        }
      }
    },
    outline: {
      label: '本页目录'
    },
    lastUpdated: {
      text: '最后更新于'
    },
    docFooter: {
      prev: '上一页',
      next: '下一页'
    },
    darkModeSwitchLabel: '外观',
    sidebarMenuLabel: '菜单',
    returnToTopLabel: '返回顶部',
    editLink: {
      pattern: 'https://github.com/ttttmr/Wechat2RSS/edit/master/:path',
      text: '在 GitHub 上编辑此页面'
    },
    logo: '/favicon.ico',
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: "免费公众号", link: "/list/" },
      { text: "私有部署", link: "/deploy/" },
      { text: "博客", link: "https://blog.xlab.app" },
    ],

    sidebar: [
      {
        text: '免费公众号',
        items: [
          { text: '合集订阅', link: '/list/' },
          { text: '完整列表', link: '/list/all' },
        ]
      },
      {
        text: '私有部署',
        items: [
          { text: '购买和定价', link: '/deploy/' },
          { text: '用户协议', link: '/deploy/agreement' },
          { text: '激活说明', link: '/deploy/active' },
          {
            text: '部署',
            items: [
              { text: '部署指南', link: '/deploy/deploy' },
              { text: '内网部署', link: '/deploy/local' },
              { text: 'Serverless代理', link: '/deploy/serverless' },
            ]
          },
          {
            text: '使用',
            items: [
              { text: '使用指南', link: '/deploy/guide' },
              { text: '服务配置', link: '/deploy/config' },
              { text: 'API参考', link: '/deploy/api' },
              { text: '常见问题', link: '/deploy/qa' },
            ]
          },
          { text: '发布记录', link: '/deploy/changelog' },
        ]
      },
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/ttttmr/wechat2rss' }
    ]
  }
})
